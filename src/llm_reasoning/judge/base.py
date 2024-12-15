from pydantic import BaseModel

from llm_reasoning.llm.base import LLM, InferenceConfig

class BaseJudge(BaseModel):
    model: LLM
    inference_config: InferenceConfig
    
    def judge(self, **kwargs):
        pass