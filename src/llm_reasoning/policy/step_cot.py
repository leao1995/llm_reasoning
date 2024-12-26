from omegaconf import OmegaConf

from llm_reasoning.task.base import State, Task, Solution
from llm_reasoning.policy.base import Policy

class StepCoT(Policy):
    env: Task
    num_chains: int
    max_depth: int
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        if policy_config.num_chains > 1:
            assert env.inference_config.temperature > 0, "greedy decoding cannot produce multiple chains"
        
        return cls(
            env=env,
            num_chains=policy_config.num_chains,
            max_depth=policy_config.depth_limit,
        )
        
    def generate_chain(self, state):
        depth = 0
        while not state.is_terminal() and depth < self.max_depth:
            action = self.env.propose_actions(state, 1)[0]
            state = self.env.transition(state, action)
            depth += 1
        
        return Solution(text=state.to_response()), state
        
    def run(self, state: State) -> tuple[list[Solution], dict]:
        solutions, final_states = [], []
        for _ in range(self.num_chains):
            solution, final_state = self.generate_chain(state)
            solutions.append(solution)
            final_states.append(final_state)
        info = {"final_state": final_states}
        
        return solutions, info
        