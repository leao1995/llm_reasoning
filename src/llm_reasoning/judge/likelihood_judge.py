import logging

from llm_reasoning.llm.base import LLM, InferenceConfig
from llm_reasoning.judge.base import BaseJudge

logger = logging.getLogger(__name__)

class LikelihoodJudge(BaseJudge):
    model: LLM
    inference_config: InferenceConfig
    
    system_prompt: str
    system_prompt_vars: list[str]
    user_prompt: str
    user_prompt_vars: list[str]
    
    candidates: list[str]
    score_idx: int
    
    def judge(self, **kwargs):
        assert all(k in kwargs for k in self.system_prompt_vars) and all(k in kwargs for k in self.user_prompt_vars)
        
        messages = [
            {"role": "system", "content": self.system_prompt.format(**{k: kwargs[k] for k in self.system_prompt_vars})},
            {"role": "user", "content": self.user_prompt.format(**{k: kwargs[k] for k in self.user_prompt_vars})},
        ]
        
        answer_probs = self.model.get_answer_probs(messages, self.candidates, self.inference_config)
        
        return answer_probs[self.score_idx]
        