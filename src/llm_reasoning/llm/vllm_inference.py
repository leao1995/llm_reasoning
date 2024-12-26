# vLLM does not support output hidden states yet
# https://github.com/vllm-project/vllm/issues/6165

import logging
import requests
import torch
import numpy as np
from copy import deepcopy
from scipy.special import logsumexp
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig
from llm_reasoning.llm.vllm_chat import truncate_generations

logger = logging.getLogger(__name__)

class vLLMInferenceModel(LLM):
    model_name: str
    port: int
    device: str
    
    _model: AutoModelForCausalLM
    _tokenizer: AutoTokenizer
    
    def __init__(self, **data):
        super().__init__(**data)
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            
    def call(self, messages: list[dict], inference_config: InferenceConfig) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": inference_config.temperature,
            "max_tokens": inference_config.max_tokens,
            "top_p": inference_config.top_p,
            "logprobs": True,
            "top_logprobs": 5,
            "add_generation_prompt": False if messages[-1]["role"] == "assistant" else True,
            "continue_final_message": True if messages[-1]["role"] == "assistant" else False,
            "chat_template": inference_config.chat_template, # custom chat template
        }
        logger.debug(f"Input message to vLLM:\n{messages}")
        response = requests.post(f"http://localhost:{self.port}/v1/chat/completions", json=payload)
        response.raise_for_status()
        output_text = response.json()["choices"][0]["message"]["content"]
        if messages[-1]["role"] == "assistant" and output_text.startswith(messages[-1]["content"]):
            output_text = output_text[len(messages[-1]["content"]):]
        finish_reason = response.json()["choices"][0]["finish_reason"]
        output_probs = response.json()["choices"][0]["logprobs"]["content"] # logprob for output tokens
        output_text, output_probs, finish_reason = truncate_generations(output_text, output_probs, finish_reason, inference_config.stop_sequences)
        logprobs = [token["logprob"] for token in output_probs]
        confidences = [np.exp(token["logprob"]-logsumexp([tok["logprob"] for tok in token["top_logprobs"]])) for token in output_probs]
        perplexity = np.exp(-np.mean(logprobs))
        # update message to obtain embeddings
        updated_messages = deepcopy(messages) # has to use deepcopy since we will change the content
        if updated_messages[-1]["role"] == "assistant": 
            updated_messages[-1]["content"] = updated_messages[-1]["content"] + output_text
        else:
            updated_messages.append({"role": "assistant", "content": output_text})
        # TODO: This may give OOM error when calling in parallel in multiple threads
        embedding = self.get_prompt_embedding(updated_messages, inference_config)
        
        return LLMResponse(
            text=output_text,
            finish_reason=finish_reason,
            perplexity=perplexity,
            logprobs=logprobs,
            confidences=confidences,
            embedding=embedding,
        )
        
    def get_prompt_embedding(self, messages: list[dict], inference_config: InferenceConfig) -> torch.Tensor:        
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
            continue_final_message=True if messages[-1]["role"] == "assistant" else False,
            chat_template=inference_config.chat_template,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(chat_prompt, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][0][-1].data.cpu().float() # last layer -> first batch idx -> last token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def _get_prompt_logprobs(self, messages: list[dict], inference_config: InferenceConfig):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1, # has to generate 1 token
            "top_p": 1.0,
            "add_generation_prompt": False if messages[-1]["role"] == "assistant" else True,
            "continue_final_message": True if messages[-1]["role"] == "assistant" else False,
            "chat_template": inference_config.chat_template, # custom chat template
            "prompt_logprobs": 1, # only return the logprob for the given token
        }
        
        response = requests.post(f"http://localhost:{self.port}/v1/chat/completions", json=payload)
        response.raise_for_status()
        
        return response.json()["prompt_logprobs"]
        
    def get_answer_probs(self, messages: list[dict], answer_candidates: list[str], inference_config: InferenceConfig):
        assert messages[-1]["role"] == "user"
        
        batch_messages = [messages] + [messages + [{"role": "assistant", "content": cand}] for cand in answer_candidates]
        with ThreadPoolExecutor() as executor:
            prompt_logprobs = list(executor.map(
                self._get_prompt_logprobs,
                batch_messages,
                [inference_config] * len(batch_messages),
            ))
        prefix_len = len(prompt_logprobs[0])
        seq_log_prob = [sum(next(iter(t.values()))['logprob'] for t in prompt_logprobs[i+1][prefix_len:]) for i in range(len(answer_candidates))]
        seq_prob = [np.exp(logprob) for logprob in seq_log_prob]
        
        return [p / sum(seq_prob) for p in seq_prob]