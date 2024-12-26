import logging
from typing import Optional
from pydantic import ConfigDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2DynamicGenertor

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class ExLlamaModel(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    draft_model_name: Optional[str]
    model_name: str
    batch_size: int
    device: str
    
    _tokenizer: AutoTokenizer
    _draft_model: Optional[ExLlamaV2]
    _draft_cache: Optional[ExLlamaV2Cache]
    _model: ExLlamaV2
    _cache: ExLlamaV2Cache
        
    _settings: ExLlamaV2Sampler.Settings
    _generator: ExLlamaV2DynamicGenertor
    
    def __init__(self, **data):
        super().__init__(**data)
        
        if self.draft_model_name is not None:
            draft_config = ExLlamaV2Config(self.draft_model_name)
            draft_config.arch_compat_overrides()
            self._draft_model = ExLlamaV2(draft_config)
            self._draft_cache = ExLlamaV2Cache(self._draft_model, lazy=True)
            self._draft_model.load_autosplit(self._draft_cache, progress=False)
        else:
            self._draft_model = None
            self._draft_cache = None
            
        config = ExLlamaV2Config(self.model_name)
        config.arch_compat_overrides()
        self._model = ExLlamaV2(config)
        self._cache = ExLlamaV2Cache(self._model, lazy=True)
        self._model.load_autosplit(self._cache, progress=False)
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        self._generator = ExLlamaV2DynamicGenertor(
            model=self._model,
            cache=self._cache,
            draft_model=self._draft_model,
            draft_cache=self._draft_cache,
            tokenizer=self._tokenizer,
            max_chunk_size=2048,
            paged=False,
        )
        self._settings = ExLlamaV2Sampler.Settings()
        self._generator.warmup()
        
    def batch_call(self, batch_messages: list[list[dict]], inference_config: InferenceConfig):
        raise NotImplementedError()
    
    def batch_inference(self, batch_input_ids: list[list[int]], inference_config: InferenceConfig):
        self._settings.temperature = inference_config.temperature
        self._settings.top_p = inference_config.top_p
        
        with torch.inference_mode():
            self._generator.generate()
        
    def encode(self, messages: list[dict], inference_config: InferenceConfig) -> list[int]:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
            continue_final_message=True if messages[-1]["role"] == "assistant" else False,
            chat_template=inference_config.chat_template, # custom chat template
        )
        
    def get_prompt_embedding(self, messages: list[dict], inference_config: InferenceConfig) -> torch.Tensor:
        assert messages[-1]["role"] == "user"
        
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False,
            continue_final_message=True,
            chat_template=inference_config.chat_template,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            self._cache.current_seq_len = 0
            outputs = self._model.forward(
                chat_prompt, 
                self._cache,
                return_last_state=True
            )
        embedding = outputs[1][0][-1].data.cpu().float() # hidden_state -> first batch idx -> last token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def get_answer_probs(self, messages: list[dict], answer_candidates: list[str], inference_config: InferenceConfig):
        raise NotImplementedError()