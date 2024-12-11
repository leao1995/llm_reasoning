import math
from omegaconf import OmegaConf
import logging
import asyncio

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
        self.V = 0
        self.Q = 0
        self.depth = depth
        
    def best_child(self, exploration_coeff=1.0):
        log_n_vertex = math.log(self.visits)

        def uct_score(node):
            exploit = node.Q
            explore = math.sqrt(2 * log_n_vertex / (1 + node.visits)) # +1 to avoid inf explore score
            return exploit + explore * exploration_coeff

        return max(self.children, key=uct_score)
    
    def add_child(self, child: 'Node'):
        self.children.append(child)
            
    def update(self):
        if self.children:
            self.V = sum(max(1, child.visits) * child.Q for child in self.children) / sum(max(1, child.visits) for child in self.children)
        self.Q = self.reward + self.V
        self.visits += 1
        
    
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
                asyncio.run(self.expand(node))
                logger.debug(f"Expanded {node.state}")
                
            # Simulation
            node = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node)
            
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
                
    async def expand(self, node):
        actions: list[Action] = self.env.propose_actions(node.state, self.breadth_limit)
        # sort based on action logprob
        actions = sorted(actions, key=lambda x: x.log_prob or 0, reverse=True)
        steps = [asyncio.create_task(self.env.step(node.state, action)) for action in actions]
        outputs = await asyncio.gather(*steps)
        
        for next_state, reward, _, _ in outputs:
            node.add_child(Node(next_state, reward, node, node.depth+1))
        
    def simulate(self, node) -> Node:
        '''
        Different from typical RL applications, here simulate will actually expand the tree to reuse the LLM calls.
        '''        
        while not node.state.is_terminal() and node.depth < self.depth_limit:
            if not node.children:
                asyncio.run(self.expand(node))
            node = node.children[0]
            
        logger.debug(f"Simulation: {node}")
            
        return node
    
    def backpropagate(self, node: Node):
        while node is not None:
            node.update()
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