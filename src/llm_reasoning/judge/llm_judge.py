import logging
from retry import retry
from pydantic import BaseModel

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig
from llm_reasoning.judge.post_processor import BaseProcessor, PostProcessingException

logger = logging.getLogger(__name__)

class LLMJudge(BaseModel):
    model: LLM
    inference_config: InferenceConfig
    
    system_prompt: str
    system_prompt_vars: list[str]
    user_prompt: str
    user_prompt_vars: list[str]
    
    post_processor: BaseProcessor
    
    def _postprocess(self, text: str):
        try:
            return self.post_processor.process(text)
        except PostProcessingException as e:
            logger.info(f"postprocessing failed: {text}")
            raise PostProcessingException(e)
    
    @retry(PostProcessingException, tries=5)
    def _judge(self, messages: list[dict]):
        llm_response: LLMResponse = self.model.call(messages, self.inference_config)
        score = self._postprocess(llm_response.text.strip())
        
        return score
    
    def judge(self, **kwargs):
        assert all(k in kwargs for k in self.system_prompt_vars) and all(k in kwargs for k in self.user_prompt_vars)
        
        messages = [
            {"role": "system", "content": self.system_prompt.format(**{k: kwargs[k] for k in self.system_prompt_vars})},
            {"role": "user", "content": self.user_prompt.format(**{k: kwargs[k] for k in self.user_prompt_vars})},
        ]
        
        try:
            return self._judge(messages)
        except PostProcessingException:
            return None