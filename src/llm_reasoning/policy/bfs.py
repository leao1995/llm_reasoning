from omegaconf import OmegaConf
from collections import deque

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
        
        
class BFS(Policy):
    env: Task
    breadth_limit: int
    depth_limit: int
    beam_size: int
    max_num_terminals: int
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        assert 'task_reward' not in env.reward_coeff, "BFS should not use task reward for searching"
        assert env.inference_config.temperature > 0, "greedy decoding cannot produce multiple reasoning chains"
        
        return cls(
            env=env,
            breadth_limit=policy_config.breadth_limit,
            depth_limit=policy_config.depth_limit,
            max_num_terminals=policy_config.max_num_terminals,
        )
        
    def run(self, state: State) -> tuple[list[Solution], dict]:
        terminal_state_dict = {}
        
        root = Node(state, 0)
        queue = deque([(root, 0.0)])
        while len(queue) > 0:
            cur_queue_size = len(queue)
            for _ in range(cur_queue_size):
                cur_node, total_reward = queue.popleft()
                if cur_node.state.is_terminal() or cur_node.depth >= self.depth_limit:
                    terminal_state_dict[cur_node.state] = total_reward
                else:
                    actions: list[Action] = self.env.propose_actions(cur_node.state, self.breadth_limit)
                    for action in actions:
                        next_state, reward, _, _ = self.env.step(cur_node.state, action)
                        next_node = Node(next_state, reward, cur_node, cur_node.depth+1)
                        cur_node.add_child(next_node)
                        queue.append((next_node, total_reward + reward))
            
            if len(terminal_state_dict) >= self.max_num_terminals:
                break
            
            # reduce the breadth to save cost
            if self.beam_size > 0:
                queue = deque(sorted(queue, key=lambda x: x[1], reverse=True)[:self.beam_size])
        
        solutions = [
            Solution(text=state.to_response(), weight=value)
            for state, value in terminal_state_dict.items()
        ]
        info = {
            "root": root,
        }
                
        return solutions, info