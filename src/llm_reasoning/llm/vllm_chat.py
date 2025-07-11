import logging
import requests
import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ThreadPoolExecutor

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class vLLMChatModel(LLM):
    model_name: str
    port: int
    
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
        
        return LLMResponse(
            text=output_text,
            finish_reason=finish_reason,
            perplexity=perplexity,
            logprobs=logprobs,
            confidences=confidences,
        )
        
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
        
    def get_answer_probs(self, messages: list[dict], answer_candidates: list[str], inference_config: InferenceConfig, normalize: bool=True):
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
        
        if normalize:
            seq_prob = [np.exp(logprob) for logprob in seq_log_prob]
            seq_prob = [p / sum(seq_prob) for p in seq_prob]
            
            return seq_prob
        
        return seq_log_prob
        
        
def truncate_generations(text: str, token_probs: list[dict], finish_reason: str, stop_sequences: list[str]):
    '''
    Truncates the input text and corresponding token probabilities at the first occurrence of any stop sequence,
    skipping leading stop sequences to avoid empty generations.
    
    Parameters:
        text (str): The input string to truncate.
        token_probs (list[dict]): A list of token probability dictionaries, each corresponding to a token in the text.
        stop_sequences (list[str]): A list of stop sequences to consider.
    
    Returns:
        tuple: A tuple containing:
            - truncated_text (str): The truncated string.
            - truncated_token_probs (list[dict]): The truncated token probabilities.
    '''
    if text.strip() != ''.join(token['token'] for token in token_probs).strip():
        logger.warning(f"tokens does not match decoded text:\n---\n{text}\n---\n{''.join(token['token'] for token in token_probs)}\n---")
    
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
            for i, token in enumerate(token_probs):
                token_length = len(token.get("token", ""))
                token_end = cumulative_length + token_length
                
                # Check if the stop sequence starts within or ends exactly at the token's end
                if cumulative_length < stop_index <= token_end:
                    truncated_token_length = i + 1  # Include this token
                    break
                
                cumulative_length = token_end
            
            # Truncate token probabilities
            token_probs = token_probs[:truncated_token_length]
    
    # Return the original text and token probabilities if no stop sequence is found
    return text, token_probs, finish_reason