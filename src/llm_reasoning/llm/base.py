from typing import Optional
from pydantic import BaseModel, ConfigDict
import torch
from concurrent.futures import ThreadPoolExecutor

class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop_sequences: list[str] = []
    chat_template: Optional[str] = None

    
class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    
    text: str
    finish_reason: str
    token_ids: Optional[list[int]] = None
    logprobs: Optional[list[float]] = None
    confidences: Optional[list[float]] = None
    perplexity: Optional[float] = None
    embedding: Optional[torch.Tensor] = None
    
    def __hash__(self):
        return hash(self.text.lower().strip())
    
    def __eq__(self, other):
        return isinstance(other, LLMResponse) and self.text.lower().strip() == other.text.lower().strip()


class LLM(BaseModel):
    model_name: str
    
    def call(self, messages: list[dict], inference_config: InferenceConfig) -> LLMResponse:
        raise NotImplementedError()
    
    def batch_call(self, batch_messages: list[list[dict]], inference_config: InferenceConfig) -> list[LLMResponse]:
        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(
                self.call,
                batch_messages,
                [inference_config] * len(batch_messages),
            ))
        return responses
    
    def inference(self, input_ids: list[int], infernce_config: InferenceConfig) -> LLMResponse:
        raise NotImplementedError()
    
    def batch_inference(self, batch_input_ids: list[list[int]], inference_config: InferenceConfig) -> list[LLMResponse]:
        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(
                self.inference,
                batch_input_ids,
                [inference_config] * len(batch_input_ids),
            ))
        return responses
    
    def encode(self, messages: list[dict], inference_config: InferenceConfig) -> list[int]:
        return None
    
    def get_prompt_embedding(self, messages: list[dict], inference_config: InferenceConfig):
        return None
        
    def get_answer_probs(self, messages: list[dict], answer_candidates: list[str], inference_config: InferenceConfig):
        return None