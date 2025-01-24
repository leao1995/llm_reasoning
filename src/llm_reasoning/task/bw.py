import json
import random
import logging
from typing import Optional
from pydantic import ConfigDict
import torch
import math
from statistics import mean
from collections import defaultdict, Counter

from llm_reasoning.task.base import Task, Solution, Action, State
from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig
from llm_reasoning.judge.base import BaseJudge
from llm_reasoning.task.blocksworld import BWExample, BWDataset, CHAT_TEMPLATE

logger = logging.getLogger(__name__)
    
ACTION_STEP_SEPARATOR = "\n"

MAX_STEP_TOKENS = 64

def get_icl_demo(prompts, examples):
    icl = prompts["intro"] + "\n".join([
        "[STATEMENT]\nAs initial conditions I have that, " + example["init"] + \
        ".\nMy goal is to have that " + example["goal"] + \
        ".\n\nMy plan is as follows:\n\n[PLAN]" + example["plan"]
        for example in examples
    ])
    icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
    return icl

class BWAction(Action):
    text: str
    finish_reason: str
    response_ids: Optional[list[int]] = None
    log_prob: Optional[float] = None
    confidence: Optional[float] = None
    embedding: Optional[torch.Tensor] = None
    

class BWState(State):
    problem: list[dict]
    problem_ids: Optional[list[int]] = None
    answer: BWExample
    trace: list[BWAction]
    embedding: Optional[torch.Tensor] = None
    max_steps: int
    
    def __str__(self):
        return "".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, BWState) and str(self) == str(other)
    
    def to_messages(self):
        messages = self.problem.copy()
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
        return len(self.trace) >= self.max_steps
    

class Blocks(Task):
    data: BWDataset
    max_steps: int
    prompts: dict
    num_shot: int
    model: LLM
    inference_config: InferenceConfig
    reward_coeff: dict[str, float]
    answer_judge: Optional[BaseJudge]
    step_judge: Optional[BaseJudge]
    
    @classmethod
    def from_config(cls, model, inference_config, task_config):
        # test data
        data = BWDataset.load(task_config.data_dir, task_config.config_file, task_config.domain_file, task_config.data_file)
        
        # prompts
        with open(task_config.prompt_file) as f:
            prompts = json.load(f)
            
        # judge for intermediate steps and final answer
        answer_judge = None
        step_judge = None
        
        # change default chat_template
        inference_config = inference_config.model_copy(update={"model_config": ConfigDict(frozen=False)}, deep=True)
        inference_config.chat_template = CHAT_TEMPLATE
        inference_config.stop_sequences += ["[PLAN END]"]
        
        return cls(
            data=data,
            max_steps=task_config.max_steps,
            prompts=prompts,
            num_shot=task_config.num_shot,
            model=model,
            inference_config=inference_config,
            reward_coeff=task_config.reward_coeff,
            answer_judge=answer_judge,
            step_judge=step_judge,
        )
        
    @property
    def size(self):
        return len(self.data.examples)
    
    def init(self, index) -> BWState:
        test_example = self.data.examples[index]
        
        # select ICL examples
        icl_examples = random.sample(self.prompts["example_pool"], self.num_shot)
        icl_template = get_icl_demo(self.prompts, icl_examples)
        problem = icl_template.replace(
            "<init_state>", test_example.init
        ).replace(
            "<goals>", test_example.goal
        ).replace(
            "<action>", ""
        )
        messages = [{"role": "user", "content": problem}]
        
        # encode problem
        problem_ids = self.model.encode(messages, self.inference_config)
        
        # get question embedding
        embedding = self.model.get_prompt_embedding(
            messages=messages,
            inference_config=InferenceConfig(chat_template=CHAT_TEMPLATE)
        )
        
        return BWState(
            problem=messages,
            problem_ids=problem_ids,
            answer=test_example,
            trace=[],
            embedding=embedding,
            max_steps=self.max_steps,
        )
        
    def transition(self, state: BWState, action: BWAction) -> BWState:
        return BWState(
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding,
            max_steps=state.max_steps,
        )
        
    async def step(self, state: BWState, action: BWAction) -> tuple[BWState, float, bool, dict]:
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
    
    def propose_actions(self, state: BWState, num_actions: int) -> list[BWAction]:
        inference_config = self.inference_config.model_copy(update={"model_config": ConfigDict(frozen=False)}, deep=True)
        inference_config.stop_sequences += [ACTION_STEP_SEPARATOR]
        inference_config.max_tokens = MAX_STEP_TOKENS
        inference_config.chat_template = CHAT_TEMPLATE
        
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
            BWAction(
                text=response.text,
                response_ids=response.token_ids[len(state.problem_ids):] if response.token_ids is not None else None,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: BWState):
        raise NotImplementedError()
    
    def eval_action(self, state: BWState, action: BWAction):
        raise NotImplementedError()
    
    def eval_solution(self, answer: BWExample, solutions: list[Solution]):
        # no valid answer
        if len(solutions) == 0:
            logger.warning(f"no valid solution is given")
            return False
        
        # only one answer
        elif len(solutions) == 1:
            generated_plan = solutions[0].text
        
        # multiple answers with weights
        elif all(solution.weight is not None for solution in solutions):
            weights = defaultdict(float)
            for solution in solutions:
                weights[solution.text] += solution.weight
            generated_plan = max(weights.items(), key=lambda x: x[1])[0]
        
        # multiple answers without weights
        else:
            answer_counts = Counter([solution.text for solution in solutions])
            generated_plan = answer_counts.most_common(1)[0][0]
        
        return answer.evaluate(self.data.data_dir, self.data.config_file, self.data.domain_file, generated_plan)