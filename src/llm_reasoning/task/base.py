from typing import Optional
from pydantic import BaseModel, ConfigDict
from datasets import Dataset
from omegaconf import OmegaConf
import torch

from llm_reasoning.llm.base import LLM, InferenceConfig


class Solution(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    
    text: str
    weight: Optional[float] = None


class Action(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    
    text: str
    log_prob: Optional[float] = None
    confidence: Optional[float] = None
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return self.text
    

class State(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    
    problem: list[dict]
    answer: str
    trace: list[Action]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        '''
        conver the state to a string for visualization
        '''
        pass
    
    def __hash__(self):
        '''
        make the state hashable to use as dict key
        '''
        pass
    
    def __eq__(self, other):
        '''
        enable comparing two states
        '''
        pass
    
    def to_messages(self):
        '''
        convert the action trajectory to messages fro LLM input
        '''
        pass
    
    def to_response(self):
        '''
        convert the action trajectory to response
        '''
        pass
    
    def is_terminal(self):
        '''
        check if a state is the terminal state
        '''
        pass


class Task(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data: Dataset
    num_shot: int
    model: LLM
    inference_config: InferenceConfig
    reward_coeff: dict[str, float]
    
    @classmethod
    def from_config(cls, model: LLM, inference_config: InferenceConfig, task_config: OmegaConf):
        '''
        initialize from config
        '''
        pass
    
    @property
    def size(self):
        '''
        the number of test examples
        '''
        pass
    
    def init(self, index) -> State:
        '''
        get the initial task state
        '''
        pass
    
    def step(self, state: State, action: Action) -> tuple[State, float, bool, dict]:
        '''
        take the next action and transition to next state
        '''
        pass
    
    def propose_actions(self, state: State, num_actions: int) -> list[Action]:
        '''
        generate the next action from prefix
        '''
        pass
    
    def eval_state(self, state: State):
        '''
        evaluate the value of the final state
        '''
        pass
    
    def eval_action(self, state: State, action: Action):
        '''
        evaluate the value of candidate action
        '''
        pass
    
    def eval_solution(self, answer: str, solutions: list[Solution]):
        '''
        if response has multiple solutions, we get the majority vote, eg SC-COT
        if weight is not None for all solutions, we select based on weights, eg MCTS
        '''
        pass
    