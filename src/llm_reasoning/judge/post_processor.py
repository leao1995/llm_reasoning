import re
from pydantic import BaseModel

class PostProcessingException(Exception):
    """Raise in case of errors during post-processing"""
    pass

class BaseProcessor(BaseModel):
    
    def process(self, text: str) -> float:
        pass

class LikertScaleProcessor(BaseProcessor):
    scales: dict[str, float]
    
    def process(self, text: str) -> float:
        if not text:
            raise PostProcessingException()
        
        score = None
        for k, v in self.scales.items():
            if re.search(k, text, re.IGNORECASE):
                score = v
                break
        
        if score is None:
            raise PostProcessingException()
        
        return score