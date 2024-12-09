from pydantic import BaseModel
from omegaconf import OmegaConf

from llm_reasoning.task.base import Task, State, Solution


class Policy(BaseModel):
    env: Task
    
    @classmethod
    def from_config(cls, env, policy_config: OmegaConf):
        pass
    
    def run(self, state: State) -> tuple[list[Solution], dict]:
        pass