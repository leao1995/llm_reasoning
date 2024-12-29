import logging
from typing import Optional
import torch
import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as vLLM, SamplingParams

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class vLLMModel(LLM):
    draft_model_name: Optional[str]
    model_name: str
    max_model_len: Optional[int]
    gpu_memory_utilization: float
    devices: list[str]
    
    _tokenizer: AutoTokenizer
    _model: vLLM
    _embed: AutoModelForCausalLM
    
    def __init__(self, **data):
        super().__init__(**data)
        
        if self.draft_model_name:
            self._model = vLLM(
                model=self.model_name,
                speculative_model=self.draft_model_name,
                num_speculative_tokens=5,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization
            )
        else:
            self._model = vLLM(model=self.model_name, max_model_len=self.max_model_len, gpu_memory_utilization=self.gpu_memory_utilization)
        
        self._embed = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(self.devices[1])
        
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
            
            # concatenate token ids to obtain embeddings
            full_input_ids = torch.tensor(list(output.prompt_token_ids) + list(token_ids)).long().to(self.devices[1]).unsqueeze(0)
            with torch.no_grad():
                embed_outputs = self._embed(full_input_ids, output_hidden_states=True)
            embedding = embed_outputs.hidden_states[-1][0][-1].data.cpu().float() # last layer -> first batch idx -> last token
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            
            response = LLMResponse(
                text=output_text,
                finish_reason=finish_reason,
                token_ids=list(outputs[0].prompt_token_ids) + list(token_ids),
                perplexity=perplexity,
                logprobs=logprobs,
                confidences=confidences,
                embedding=embedding
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
            
            # concatenate token ids to obtain embeddings
            full_input_ids = torch.tensor(list(output.prompt_token_ids) + list(token_ids)).long().to(self.devices[1]).unsqueeze(0)
            with torch.no_grad():
                embed_outputs = self._embed(full_input_ids, output_hidden_states=True)
            embedding = embed_outputs.hidden_states[-1][0][-1].data.cpu().float() # last layer -> first batch idx -> last token
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            
            response = LLMResponse(
                text=output_text,
                finish_reason=finish_reason,
                token_ids=list(output.prompt_token_ids) + list(token_ids),
                perplexity=perplexity,
                logprobs=logprobs,
                confidences=confidences,
                embedding=embedding
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
        assert messages[-1]["role"] == "user"
        
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False,
            continue_final_message=True,
            chat_template=inference_config.chat_template,
            return_tensors="pt"
        ).to(self.devices[1])
        with torch.no_grad():
            outputs = self._embed(chat_prompt, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][0][-1].data.cpu().float() # last layer -> first batch idx -> last token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
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
        seq_log_prob = [sum(next(iter(t.values())).logprob for t in prompt_logprobs[i+1][prefix_len:]) for i in range(len(answer_candidates))]
        seq_prob = [np.exp(logprob) for logprob in seq_log_prob]
        
        return [p / sum(seq_prob) for p in seq_prob]
    
def truncate_generations(text: str, token_ids: list[int], token_probs: list[dict], finish_reason: str, stop_sequences: list[str]):
    if text.strip() != ''.join(p[i].decoded_token for i, p in zip(token_ids, token_probs)).strip():
        logger.warning(f"tokens does not match decoded text:\n---\n{text}\n---\n{''.join(p[i].decoded_token for i, p in zip(token_ids, token_probs))}\n---")
        
    # Strip leading whitespace from text
    stripped_text = text.lstrip()
    leading_whitespace_len = len(text) - len(stripped_text)
    
    for stop_seq in stop_sequences:
        # Skip leading stop sequences
        index = leading_whitespace_len
        while text.startswith(stop_seq, index):
            index += len(stop_seq)
            
        # Find the next occurrence of the stop sequence
        stop_index = text.find(stop_seq, index)
        if stop_index != -1:
            # Truncate the text at the first valid stop sequence
            text = text[:stop_index]
            finish_reason = "truncation"
            
            # Calculate the corresponding token length
            cumulative_length = 0
            truncated_token_length = 0
            for i, (token, probs) in enumerate(zip(token_ids, token_probs)):
                token_length = len(probs[token].decoded_token)
                token_end = cumulative_length + token_length
                
                # Check if the stop sequence starts within or ends exactly at the token's end
                if cumulative_length < stop_index <= token_end:
                    truncated_token_length = i + 1  # Include this token
                    break
                
                cumulative_length = token_end
                
            # Truncate token probabilities
            token_ids = token_ids[:truncated_token_length]
            token_probs = token_probs[:truncated_token_length]
            
    # Return the original text and token probabilities if no stop sequence is found
    return text, token_ids, token_probs, finish_reason