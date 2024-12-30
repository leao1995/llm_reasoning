import math
import logging
import asyncio
from functools import partial
from omegaconf import OmegaConf

from llm_reasoning.task.base import State, Action, Task, Solution
from llm_reasoning.policy.base import Policy

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, step_fn, state, reward, info, parent=None, depth=0):
        self.step_fn = step_fn
        self.state = state
        self.reward = reward # rewards from the action that leads to this state
        self.info = info # info from the action that leads to this state
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
        if 'task_reward' in env.reward_coeff: 
            logger.warning("MCTS should not use task reward for searching, make sure you know what you are doing.")
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
        
        root = Node(None, state, 0, {})
        
        for i in range(self.mcts_iterations):
            logger.debug(f"Iteration: {i}")
            # Selection
            node = self.select(root)
            logger.debug(f"Selection")

            # Expansion
            if not node.state.is_terminal() and node.depth < self.depth_limit:
                self.expand(node)
                logger.debug(f"Expanded")
                
            # Simulation
            node = self.simulate(node)
            logger.debug(f"Simulation")
            
            # Backpropagation
            asyncio.run(self.backpropagate(node))
            
        terminal_state_dict = self.traverse(root)
        logger.debug(f"terminal_states: {terminal_state_dict}")
        
        solutions = [
            Solution(text=state.to_response(), weight=value)
            for state, value in terminal_state_dict.items()
        ]
        info = {
            "root": root,
        }
        
        return solutions, info
            
    def select(self, node):
        while node.children:
            node = node.best_child(self.exploration_coeff)
            
        return node
        
    def expand(self, node):
        actions: list[Action] = self.env.propose_actions(node.state, self.breadth_limit)
        # sort based on action logprob
        actions = sorted(actions, key=lambda x: x.log_prob or 0, reverse=True)
        for action in actions:
            next_state = self.env.transition(node.state, action) # state is needed to check stop condition
            step_fn = partial(self.env.step, node.state, action) # This will create a coroutine, not a task, dealy execute to backpropagation
            node.add_child(Node(step_fn, next_state, None, None, node, node.depth+1))
        
    def simulate(self, node) -> Node:
        '''
        Different from typical RL applications, here simulate will actually expand the tree to reuse the LLM calls.
        '''
        while not node.state.is_terminal() and node.depth < self.depth_limit:
            if not node.children:
                self.expand(node)
            node = node.children[0]
            
        return node
    
    async def backpropagate(self, node: Node):
        # execute the step function
        nodes_to_execute = []
        cur_node = node
        while cur_node is not None:
            if cur_node.reward is None and cur_node.step_fn is not None:
                nodes_to_execute.append(cur_node)
            cur_node = cur_node.parent
        outputs = await asyncio.gather(*[n.step_fn() for n in nodes_to_execute])
        logger.debug(f"Execute step function: {len(nodes_to_execute)}")
        
        # update reward
        for n, (next_state, reward, _, info) in zip(nodes_to_execute, outputs):
            n.state = next_state
            n.reward = reward
            n.info = info
        
        # backpropagation
        cur_node = node
        while cur_node is not None:
            cur_node.update()
            cur_node = cur_node.parent
        logger.debug("Backpropagate")
            
    def traverse(self, node) -> dict[State, float]:
        terminal_state_dict = {}
        
        def dfs(node, total_reward):
            if node.state.is_terminal():
                terminal_state_dict[node.state] = total_reward
                return
            # cleanup step functions for all children
            for child in node.children:
                child.step_fn = None
            # traverse only visited children
            visited_childred = [child for child in node.children if child.visits > 0]
            for child in visited_childred:
                dfs(child, total_reward + child.reward)
                
        dfs(node, 0)
            
        return terminal_state_dict