import logging
from retry import retry
import litellm
from litellm.exceptions import APIConnectionError

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class APIModel(LLM):
    model_name: str
    
    @retry(APIConnectionError, tries=10, delay=0.1, backoff=3)
    def call(self, messages: list[dict], inference_config: InferenceConfig) -> LLMResponse:
        if inference_config.chat_template is not None:
            raise NotImplementedError("custom chat template is not supported for api models")
        
        ## assistant prefill: note only certain models support this feature
        ## https://docs.litellm.ai/docs/completion/prefix
        if messages[-1]["role"] == "assistant":
            messages[-1]["prefix"] = True
        
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            max_tokens=inference_config.max_tokens,
        )
        text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        text, finish_reason = truncate_generations(text, finish_reason, inference_config.stop_sequences)
        
        return LLMResponse(
            text=text,
            finish_reason=finish_reason,
        )
        
def truncate_generations(text: str, finish_reason: str, stop_sequences: list[str]):
    '''
    Truncates the input text at the first occurrence of any stop sequence,
    skipping leading stop sequences to avoid empty generations.
    
    Parameters:
        text (str): The input string to truncate.
        stop_sequences (list[str]): A list of stop sequences to consider.
    
    Returns:
        str: The truncated string.
    '''
    text = text.lstrip()
    for stop_seq in stop_sequences:
        # Find the first occurrence of the stop sequence after leading occurrences
        index = 0
        while text.startswith(stop_seq, index):
            index += len(stop_seq)  # Skip over leading stop sequences
        
        # Find the next occurrence of the stop sequence
        stop_index = text.find(stop_seq, index)
        if stop_index != -1:
            text = text[:stop_index]
            finish_reason = "truncation"
    
    # Return the original text if no stop sequence is found
    return text, finish_reason