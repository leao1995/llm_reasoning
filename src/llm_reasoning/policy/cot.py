from omegaconf import OmegaConf

from llm_reasoning.task.base import State, Action, Task, Solution
from llm_reasoning.policy.base import Policy
from llm_reasoning.llm.base import LLMResponse

class CoT(Policy):
    env: Task
    num_chains: int
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        if policy_config.num_chains > 1:
            assert env.inference_config.temperature > 0, "greedy decoding cannot produce multiple chains"
        
        return cls(
            env=env,
            num_chains=policy_config.num_chains,
        )
        
    def run(self, state: State) -> tuple[list[Solution], dict]:
        messages = state.to_messages()
        llm_responses: list[LLMResponse] = self.env.model.batch_call(
            batch_messages=[messages] * self.num_chains,
            inference_config=self.env.inference_config,
        )
        
        solutions = [
            Solution(text=response.text)
            for response in llm_responses
        ]
        info = {}
        
        return solutions, info