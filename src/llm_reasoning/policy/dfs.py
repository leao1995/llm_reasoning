from omegaconf import OmegaConf

from llm_reasoning.task.base import State, Action, Task, Solution
from llm_reasoning.policy.base import Policy


class Node:
    def __init__(self, state, reward, parent=None, depth=0):
        self.state = state
        self.reward = reward
        self.parent = parent
        self.children = []
        self.depth = depth
        
    def add_child(self, child: 'Node'):
        self.children.append(child)


class DFS(Policy):
    env: Task
    breadth_limit: int
    depth_limit: int
    max_num_terminals: int
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        assert 'task_reward' not in env.reward_coeff, "DFS should not use task reward for searching"
        assert env.inference_config.temperature > 0, "greedy decoding cannot produce multiple reasoning chains"
        
        return cls(
            env=env,
            breadth_limit=policy_config.breadth_limit,
            depth_limit=policy_config.depth_limit,
            max_num_terminals=policy_config.max_num_terminals,
        )
    
    def run(self, state: State) -> tuple[list[Solution], dict]:
        terminal_state_dict = {}
        
        def dfs(cur_node: Node, total_reward: float):
            if len(terminal_state_dict) >= self.max_num_terminals:
                return 
            
            if cur_node.state.is_terminal() or cur_node.depth >= self.depth_limit:
                terminal_state_dict[cur_node.state] = total_reward
                return
            
            actions: list[Action] = self.env.propose_actions(cur_node.state, self.breadth_limit)
            # sort based on action logprob
            actions = sorted(actions, key=lambda x: x.log_prob or 0, reverse=True)
            # depth first search
            for action in actions:
                next_state, reward, _, _ = self.env.step(cur_node.state, action)
                next_node = Node(next_state, reward, cur_node, cur_node.depth+1)
                cur_node.add_child(next_node)
                dfs(next_node, total_reward + reward)
        
        root = Node(state, 0)
        dfs(root, 0)
        
        solutions = [
            Solution(text=state.to_response(), weight=value)
            for state, value in terminal_state_dict.items()
        ]
        info = {
            "root": root
        }
        
        return solutions, info
        