import logging
from typing import Optional
from pydantic import ConfigDict

from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2DynamicGenertor

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class ExLlamaModel(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    draft_model_name: Optional[str]
    model_name: str
    batch_size: int
    device: str
    
    _draft_model: Optional[ExLlamaV2]
    _draft_cache: Optional[ExLlamaV2Cache]
    _model: ExLlamaV2
    _cache: ExLlamaV2Cache
    
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
            
        
