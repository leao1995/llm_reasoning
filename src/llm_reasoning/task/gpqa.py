import re
import logging
from typing import Optional
from pydantic import ConfigDict
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from collections import Counter, defaultdict
import torch
import math
from statistics import mean

from llm_reasoning.task.base import Task, Solution, Action, State
from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig
from llm_reasoning.judge.base import BaseJudge
from llm_reasoning.judge.llm_judge import LLMJudge
from llm_reasoning.judge.post_processor import LikertScaleProcessor
from llm_reasoning.judge.likelihood_judge import LikelihoodJudge

logger = logging.getLogger(__name__)

# adapted from opencompass
IN_CONTEXT_EXAMPLES = []

PROMPT_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ACTION_STEP_SEPARATOR = "\n"

MAX_STEP_TOKENS = 64

ANSWER_SELFEVAL_SYSTEM_PROMPT = ""

ANSWER_SELFEVAL_USER_PROMPT = ""

STEP_SELFEVAL_SYSTEM_PROMPT = ""

STEP_SELFEVAL_USER_PROMPT = ""

def extract_answer(text: str) -> str:
    ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([A-D])'
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None

def check_answer_correctness(ground_truth: str, response: str):
    try:
        if ground_truth == response:
            return True
    except Exception:
        pass
    
    return False

class GPQAAction(Action):
    text: str
    finish_reason: str
    response_ids: Optional[list[int]] = None
    log_prob: Optional[float] = None
    confidence: Optional[float] = None
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return self.text
    
    def is_final_action(self):
        return self.finish_reason == "stop"
    

class GPQAState(State):
    problem: list[dict]
    problem_ids: Optional[list[int]] = None
    answer: str
    trace: list[GPQAAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "\n\n".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, GPQAState) and str(self) == str(other)
    
    def to_messages(self):
        messages = self.problem.copy() # copy to avoid changing the problem
        if self.trace:
            assistant_prefill = self.to_response() + ACTION_STEP_SEPARATOR
            messages.append({"role": "assistant", "content": assistant_prefill})
            
        return messages
    
    def to_input_ids(self):
        input_ids = self.problem_ids.copy()
        if self.trace:
            input_ids += self.trace[-1].response_ids
            
        return input_ids
    
    def to_response(self):
        return ACTION_STEP_SEPARATOR.join([action.text for action in self.trace]).lstrip()
    
    def is_terminal(self):
        if not self.trace:
            return False
        return bool(re.search(r"ANSWER:", self.trace[-1].text)) or self.trace[-1].is_final_action()


class GPQA(Task):
    data: Dataset
    num_shot: int
    model: LLM
    inference_config: InferenceConfig
    reward_coeff: dict[str, float]
    answer_judge: Optional[BaseJudge]
    step_judge: Optional[BaseJudge]
    
    @classmethod
    def from_config(cls, model: LLM, inference_config: InferenceConfig, task_config: OmegaConf):
        assert task_config.num_shot == 0
        
        # test data
        if task_config.split == "train":
            data = load_dataset("jeggers/gpqa_formatted", "main", split="train")
            exclude = load_dataset("jeggers/gpqa_formatted", "diamond", split="train")
            exclude_questions = set(exclude["Question"])
            data = data.filter(lambda x: x["Question"] not in exclude_questions)
        else:
            data = load_dataset("jeggers/gpqa_formatted", "diamond", split="train")
        if task_config.max_num_instances > 0:
            data = data.select(range(min(task_config.max_num_instances, len(data))))
            
        # judge for intermediate steps and final answer
        answer_judge = None
        step_judge = None
        
        return cls(
            data=data,
            num_shot=0,
            model=model,
            inference_config=inference_config,
            reward_coeff=task_config.reward_coeff,
            answer_judge=answer_judge,
            step_judge=step_judge,
        )
        
    @property
    def size(self):
        return len(self.data)
    
    def init(self, index) -> GPQAState:
        messages = []
        
        # select test examples
        test_example = self.data[index]
        test_prompt = PROMPT_TEMPLATE.format(
            Question=test_example["Question"].strip(),
            A=test_example["options"][0].strip(),
            B=test_example["options"][1].strip(),
            C=test_example["options"][2].strip(),
            D=test_example["options"][3].strip(),
        )
        messages.append({"role": "user", "content": test_prompt})
        
        # encode problem
        problem_ids = self.model.encode(messages, self.inference_config)
        
        # create init state
        init_state = GPQAState(
            problem=messages,
            problem_ids=problem_ids,
            answer="ABCD"[test_example["answer"]],
            trace=[],
            embedding=self.model.get_prompt_embedding(messages=messages, inference_config=InferenceConfig())
        )
        
        return init_state
    
    def transition(self, state: GPQAState, action: GPQAAction) -> GPQAState:
        return GPQAState(
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding
        )
        
    async def step(self, state: GPQAState, action: GPQAAction) -> tuple[GPQAState, float, bool, dict]:
        next_state = self.transition(state, action)
        
        done = next_state.is_terminal()
        
        info = {}
        
        reward = 0.0
        if 'action_logprob' in self.reward_coeff and action.log_prob is not None:
            reward += math.exp(action.log_prob) * self.reward_coeff["action_logprob"] # exp to avoid negative reward
            info["action_logprob"] = math.exp(action.log_prob)
        if 'action_confidence' in self.reward_coeff and action.confidence is not None:
            reward += action.confidence * self.reward_coeff["action_confidence"]
            info["action_confidence"] = action.confidence
        if 'action_quality' in self.reward_coeff:
            action_quality = self.eval_action(state, action)
            reward += action_quality * self.reward_coeff["action_quality"]
            info["action_quality"] = action_quality
        if done and 'answer_quality' in self.reward_coeff:
            answer_quality = self.eval_state(next_state)
            reward += answer_quality * self.reward_coeff["answer_quality"]
            info["answer_quality"] = answer_quality
        if done and 'task_reward' in self.reward_coeff:
            solution = Solution(text=next_state.to_response())
            task_reward = self.eval_solution(state.answer, [solution])
            reward += task_reward * self.reward_coeff["task_reward"]
            info["task_reward"] = task_reward
            
        return next_state, reward, done, info
    
    def propose_actions(self, state: GPQAState, num_actions: int) -> list[GPQAAction]:
        inference_config = self.inference_config.model_copy(update={"model_config": ConfigDict(frozen=False)}, deep=True)
        inference_config.stop_sequences += [ACTION_STEP_SEPARATOR]
        inference_config.max_tokens = MAX_STEP_TOKENS
        
        if state.problem_ids is not None:
            input_ids = state.to_input_ids()
            llm_responses: list[LLMResponse] = self.model.batch_inference(
                batch_input_ids=[input_ids] * num_actions,
                inference_config=inference_config,
            )
        else:
            messages = state.to_messages()
            llm_responses: list[LLMResponse] = self.model.batch_call(
                batch_messages=[messages] * num_actions,
                inference_config=inference_config,
            )
            
        llm_responses = list(dict.fromkeys(llm_responses)) # remove duplicated responses
        
        return [
            GPQAAction(
                text=response.text,
                response_ids=response.token_ids[len(state.problem_ids):] if response.token_ids is not None else None,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: GPQAState):
        problem = state.problem[-1]["content"]
        response = state.to_response()
        
        score = self.answer_judge.judge(QUESTION=problem, ANSWER=response)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_action(self, state: GPQAState, action: GPQAAction):
        problem = state.problem[-1]["content"]
        previous_response = state.to_response()
        current_step = action.text
        
        score = self.step_judge.judge(QUESTION=problem, PREVIOUS_STEPS=previous_response, CURRENT_STEP=current_step)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_solution(self, answer: str, solutions: list[Solution]):
        ground_truth = answer
        
        # no valid answer
        if len(solutions) == 0:
            logger.warning(f"no valid solution is given")
            return False
        
        # only one answer
        elif len(solutions) == 1:
            response = extract_answer(solutions[0].text)
            
        # multiple answers with weights
        elif all(solution.weight is not None for solution in solutions):
            weights = defaultdict(float)
            for solution in solutions:
                weights[extract_answer(solution.text)] += solution.weight
            response = max(weights.items(), key=lambda x: x[1])[0]
            
        # multiple answers without weights
        else:
            answer_counts = Counter([extract_answer(solution.text) for solution in solutions])
            response = answer_counts.most_common(1)[0][0]
            
        return check_answer_correctness(ground_truth, response)