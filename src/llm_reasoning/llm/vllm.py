import requests
import numpy as np
from scipy.special import logsumexp

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

class vLLMModel(LLM):
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
        }
        response = requests.post(f"http://localhost:{self.port}/v1/chat/completions", json=payload)
        response.raise_for_status()
        output_text = response.json()["choices"][0]["message"]["content"]
        if messages[-1]["role"] == "assistant" and output_text.startswith(messages[-1]["content"]):
            output_text = output_text[len(messages[-1]["content"]):]
        output_probs = response.json()["choices"][0]["logprobs"]["content"] # logprob for output tokens
        output_text, output_probs = truncate_generations(output_text, output_probs, inference_config.stop_sequences)
        logprobs = [token["logprob"] for token in output_probs]
        confidences = [np.exp(token["logprob"]-logsumexp([tok["logprob"] for tok in token["top_logprobs"]])) for token in output_probs]
        perplexity = np.exp(-np.mean(logprobs))
        
        return LLMResponse(
            text=output_text,
            perplexity=perplexity,
            logprobs=logprobs,
            confidences=confidences,
        )
        
def truncate_generations(text: str, token_probs: list[dict], stop_sequences: list[str]):
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
    assert text == ''.join(token['token'] for token in token_probs)
    
    # Strip leading whitespace from text
    stripped_text = text.lstrip()
    leading_whitespace_len = len(text) - len(stripped_text)
    
    # Adjust token_start_index by summing actual token contributions in output text
    token_start_index = 0
    cumulative_length = 0
    for i, token in enumerate(token_probs):
        # Extract the token text as it appears in the output
        token_length = len(token.get("token", ""))
        
        # Skip tokens contributing to the leading whitespace
        cumulative_length += token_length
        if cumulative_length > leading_whitespace_len:
            token_start_index = i
            break
            
    # Update the text and tokens after removing leading whitespace
    text = stripped_text
    token_probs = token_probs[token_start_index:]
    
    for stop_seq in stop_sequences:
        # Skip leading stop sequences
        index = 0
        while text.startswith(stop_seq, index):
            index += len(stop_seq)
            
        # Find the next occurrence of the stop sequence
        stop_index = text.find(stop_seq, index)
        if stop_index != -1:
            # Truncate the text at the first valid stop sequence
            text = text[:stop_index]
            
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
    return text, token_probs