import logging
import random
import torch
from typing import Optional
from datasets import load_dataset, load_from_disk

from llm_reasoning.llm.base import LLM, InferenceConfig
from llm_reasoning.task.base import Task, State
from llm_reasoning.task.gsm8k import GSM8K
from llm_reasoning.task.math import Math

logger = logging.getLogger(__name__)

class MetaState(State):    
    task: str
    state: State
    
    problem: list[dict]
    problem_ids: Optional[list[int]]
    answer: str
    trace: list
    embedding: Optional[torch.Tensor]

class MetaMath(Task):
    data: dict
    num_shot: int
    model: LLM
    inference_config: InferenceConfig
    reward_coeff: dict[str, float]
    
    @classmethod
    def from_config(cls, model, inference_config, task_config):
        assert task_config.split == "train"
        # gsm8k
        gsm8k_data = load_dataset("gsm8k", "main", split="train")
        gsm8k_task = GSM8K(
            data=gsm8k_data,
            num_shot=4,
            model=model,
            inference_config=inference_config,
            reward_coeff=task_config.reward_coeff,
            answer_judge=None,
            step_judge=None
        )
        # math
        # math_data = load_dataset("hendrycks/competition_math", trust_remote_code=True, split="train")
        math_data = load_from_disk("data/math/train")
        math_data = math_data.shuffle(seed=42)
        math_task = Math(
            data=math_data,
            num_shot=4,
            model=model,
            inference_config=inference_config,
            reward_coeff=task_config.reward_coeff,
            answer_judge=None,
            step_judge=None
        )

        return cls(
            data={"gsm8k": gsm8k_task, "math": math_task},
            num_shot=-1,
            model=model,
            inference_config=inference_config,
            reward_coeff=task_config.reward_coeff,
        )
        
    @property
    def size(self):
        return sum([self.data[task].size for task in self.data])
    
    def init(self, index) -> MetaState:
        cur_task = random.choice(list(self.data.keys()))
        task_idx = random.choice(range(self.data[cur_task].size))
        state = self.data[cur_task].init(task_idx)
        
        return MetaState(
            task=cur_task,
            state=state,
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace,
            embedding=state.embedding
        )
    
    def transition(self, state: MetaState, action) -> MetaState:
        raise NotImplementedError()
    
    async def step(self, state: MetaState, action):
        next_state, reward, done, info = await self.data[state.task].step(state.state, action)
        meta_state = MetaState(
            task=state.task, 
            state=next_state,
            problem=next_state.problem,
            problem_ids=next_state.problem_ids,
            answer=next_state.answer,
            trace=next_state.trace,
            embedding=next_state.embedding
        )
        return meta_state, reward, done, info
    
    def propose_actions(self, state: MetaState, num_actions):
        return self.data[state.task].propose_actions(state.state, num_actions)
    
    def eval_state(self, state):
        return NotImplementedError()
    
    def eval_action(self, state, action):
        return NotImplementedError()
    
    def eval_solution(self, answer, solutions):
        return NotImplementedError()