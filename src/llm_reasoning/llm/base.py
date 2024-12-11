from typing import Optional
from pydantic import BaseModel, ConfigDict
import torch
from tqdm import tqdm
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
            responses = list(
                tqdm(
                    executor.map(
                        self.call,
                        batch_messages,
                        [inference_config] * len(batch_messages),
                    ),
                    total=len(batch_messages),
                    desc="llm calls",
                )
            )
        return responses
    
    def get_prompt_embedding(self, messages: list[dict]):
        return None
        
        