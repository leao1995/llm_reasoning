import math
from omegaconf import OmegaConf
import logging

from llm_reasoning.task.base import State, Action, Task, Solution
from llm_reasoning.policy.base import Policy

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state, reward, parent=None, depth=0):
        self.state = state
        self.reward = reward
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = depth
        
    def best_child(self, exploration_coeff=1.0):
        log_n_vertex = math.log(self.visits)

        def ucb_score(node):
            if node.visits == 0:
                return float('inf')
            exploit = node.value / node.visits
            explore = math.sqrt(2 * log_n_vertex / node.visits)
            return exploit + explore * exploration_coeff

        return max(self.children, key=ucb_score)
    
    def add_children(self, next_states: list[State], rewards: list[float]):
        self.children = [Node(state, reward, self, self.depth+1) for state, reward in zip(next_states, rewards)]
            
    def update(self, value):
        self.visits += 1
        self.value += value
        
    
class MCTS(Policy):
    env: Task
    mcts_iterations: int
    breadth_limit: int
    depth_limit: int
    exploration_coeff: float = 1.0
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        assert 'task_reward' not in env.reward_coeff, "MCTS should not use task reward for searching"
        assert env.inference_config.temperature > 0, "greedy decoding cannot produce multiple reasoning chains"
        
        return cls(
            env=env,
            mcts_iterations=policy_config.mcts_iterations,
            breadth_limit=policy_config.breadth_limit,
            depth_limit=policy_config.depth_limit,
            exploration_coeff=policy_config.exploration_coeff
        )
    
    def run(self, state: State) -> tuple[list[Solution], dict]:
        logger.debug("Start MCTS")
        
        root = Node(state, 0)
        
        selections = []
        
        for i in range(self.mcts_iterations):
            logger.debug(f"Iteration: {i}")
            # Selection
            node = self.select(root)
            selections.append(node)
            logger.debug(f"Selection: {node.state}")

            # Expansion
            if not node.state.is_terminal() and node.depth < self.depth_limit:
                self.expand(node)
                logger.debug(f"Expanded {node.state}")
                
            # Simulation
            reward = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node, reward)
            
        terminal_state_dict = self.traverse(root)
        logger.debug(f"terminal_states: {terminal_state_dict}")
        
        solutions = [
            Solution(text=state.to_response(), weight=value)
            for state, value in terminal_state_dict.items()
        ]
        info = {
            "root": root,
            "selections": selections
        }
        
        return solutions, info
            
    def select(self, node):
        while node.children:
            node = node.best_child(self.exploration_coeff)
            
        return node
                
    def expand(self, node):
        actions: list[Action] = self.env.propose_actions(node.state, self.breadth_limit)
        next_states: list[State] = []
        rewards = []
        for action in actions:
            next_state, reward, _, _ = self.env.step(node.state, action)
            next_states.append(next_state)
            rewards.append(reward)
        node.add_children(next_states, rewards)
        
    def simulate(self, node) -> float:
        total_reward = 0
        state = node.state
        depth = node.depth
        done = False
        
        while not done and depth < self.depth_limit:
            action: Action = self.env.propose_actions(state, 1)[0]
            state, reward, done, _ = self.env.step(state, action)
            total_reward += reward
            depth += 1
            
        logger.debug(f"Simulation: {state}")
            
        return total_reward
    
    def backpropagate(self, node, value):
        while node is not None:
            node.update(value)
            value += node.reward
            node = node.parent
            
    def traverse(self, node) -> dict[State, float]:
        terminal_state_dict = {}
        
        def dfs(node, total_reward):
            if node.state.is_terminal():
                terminal_state_dict[node.state] = total_reward
                return
            for child in node.children:
                dfs(child, total_reward + child.reward)
                
        dfs(node, 0)
            
        return terminal_state_dict