import logging
from typing import Optional
import torch
import numpy as np
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as vLLM, SamplingParams
from llm_reasoning.llm.vllm import truncate_generations

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class qvLLMModel(LLM):
    draft_model_name: Optional[str]
    model_name: str
    max_model_len: Optional[int]
    gpu_memory_utilization: float
    devices: list[str]
    
    _tokenizer: AutoTokenizer
    _model: vLLM
    
    def __init__(self, **data):
        super().__init__(**data)
        
        if self.draft_model_name:
            self._model = vLLM(
                model=self.model_name,
                speculative_model=self.draft_model_name,
                num_speculative_tokens=5,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                quantization="bitsandbytes", 
                load_format="bitsandbytes",
                tensor_parallel_size=2,
            )
        else:
            self._model = vLLM(
                model=self.model_name, 
                max_model_len=self.max_model_len, 
                gpu_memory_utilization=self.gpu_memory_utilization,
                quantization="bitsandbytes", 
                load_format="bitsandbytes",
                tensor_parallel_size=2,
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            
    def batch_call(self, batch_messages: list[list[dict]], inference_config: InferenceConfig) -> list[LLMResponse]:
        sampling_params = SamplingParams(
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            max_tokens=inference_config.max_tokens,
            logprobs=5,
        )
        is_continue = [messages[-1]["role"] == "assistant" for messages in batch_messages]
        assert len(set(is_continue)) == 1
        
        outputs = self._model.chat(
            batch_messages,
            sampling_params,
            use_tqdm=False,
            chat_template=inference_config.chat_template,
            add_generation_prompt=not all(is_continue),
            continue_final_message=all(is_continue)
        )
        
        responses: list[LLMResponse] = []
        for output in outputs:
            output_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason
            output_probs = output.outputs[0].logprobs
            
            output_text, token_ids, output_probs, finish_reason = truncate_generations(output_text, token_ids, output_probs, finish_reason, inference_config.stop_sequences)

            logprobs = [p[i].logprob for i, p in zip(token_ids, output_probs)]
            confidences = [np.exp(p[i].logprob - logsumexp([p[tok].logprob for tok in p])) for i, p in zip(token_ids, output_probs)]
            perplexity = np.exp(-np.mean(logprobs))
            
            response = LLMResponse(
                text=output_text,
                finish_reason=finish_reason,
                token_ids=list(outputs[0].prompt_token_ids) + list(token_ids),
                perplexity=perplexity,
                logprobs=logprobs,
                confidences=confidences,
                embedding=None
            )
            responses.append(response)
            
        return responses
        
    def batch_inference(self, batch_input_ids: list[list[int]], inference_config: InferenceConfig) -> list[LLMResponse]:
        sampling_params = SamplingParams(
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            max_tokens=inference_config.max_tokens,
            logprobs=5,
        )
        outputs = self._model.generate(
            prompt_token_ids=batch_input_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        responses: list[LLMResponse] = []
        for output in outputs:
            output_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason
            output_probs = output.outputs[0].logprobs
            
            output_text, token_ids, output_probs, finish_reason = truncate_generations(output_text, token_ids, output_probs, finish_reason, inference_config.stop_sequences)

            logprobs = [p[i].logprob for i, p in zip(token_ids, output_probs)]
            confidences = [np.exp(p[i].logprob - logsumexp([p[tok].logprob for tok in p])) for i, p in zip(token_ids, output_probs)]
            perplexity = np.exp(-np.mean(logprobs))
            
            response = LLMResponse(
                text=output_text,
                finish_reason=finish_reason,
                token_ids=list(output.prompt_token_ids) + list(token_ids),
                perplexity=perplexity,
                logprobs=logprobs,
                confidences=confidences,
                embedding=None
            )
            responses.append(response)
            
        return responses
        
    def encode(self, messages: list[dict], inference_config: InferenceConfig) -> list[int]:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
            continue_final_message=True if messages[-1]["role"] == "assistant" else False,
            chat_template=inference_config.chat_template, # custom chat template
        )
        
    def get_prompt_embedding(self, messages: list[dict], inference_config: InferenceConfig) -> torch.Tensor:
        return None
    
    
    def _get_prompt_logprobs(self, messages: list[dict], inference_config: InferenceConfig):
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
        )
        outputs = self._model.chat(
            messages,
            sampling_params,
            use_tqdm=False,
            chat_template=inference_config.chat_template,
            add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
            continue_final_message=True if messages[-1]["role"] == "assistant" else False
        )
        
        return outputs[0].prompt_logprobs        
    
    def get_answer_probs(self, messages: list[dict], answer_candidates: list[str], inference_config: InferenceConfig, normalize: bool=True):
        assert messages[-1]["role"] == "user"
        
        batch_messages = [messages] + [messages + [{"role": "assistant", "content": cand}] for cand in answer_candidates]
        prompt_logprobs = [self._get_prompt_logprobs(messages, inference_config) for messages in batch_messages]
        prefix_len = len(prompt_logprobs[0])
        seq_log_prob = [sum(next(iter(t.values())).logprob for t in prompt_logprobs[i+1][prefix_len:]) for i in range(len(answer_candidates))]
        
        if normalize:
            seq_prob = [np.exp(logprob) for logprob in seq_log_prob]
            seq_prob = [p / sum(seq_prob) for p in seq_prob]
            
            return seq_prob
        
        return seq_log_prob