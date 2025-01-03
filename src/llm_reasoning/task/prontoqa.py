import re
import os
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

IN_CONTEXT_EXAMPLES = [
    {
        'question': 'Q: Animals are multicellular. Carnivores are carnivorous. Mammals are vertebrates. Each mammal is not cold-blooded. Felines are carnivores. Each carnivore is a mammal. Each cat is a feline. Vertebrates are animals. Each snake is cold-blooded. Fae is a feline. True or false: Fae is not cold-blooded.\nA:',
        'answer': 'Fae is a feline.\nFelines are carnivores. So Fae is a carnivore.\nEach carnivore is a mammal. So Fae is a mammal.\nEach mammal is not cold-blooded. So Fae is not cold-blooded.\nThe answer is true.'
    },
    {
        'question': 'Q: Natural numbers are positive. Real numbers are real. Each real number is a number. Every prime number is a natural number. Imaginary numbers are not real. Each integer is a real number. Prime numbers are prime. Mersenne primes are prime numbers. Mersenne primes are not composite. Natural numbers are integers. 127 is a natural number. True or false: 127 is not real.\nA:',
        'answer': '127 is a natural number.\nNatural numbers are integers. So 127 is an integer.\nEach integer is a real number. So 127 is a real number.\nReal numbers are real. So 127 is real.\nThe answer is false.'
    },
    {
        'question': 'Q: Every lepidopteran is an insect. Spiders are not six-legged. Each insect is an arthropod. Each invertebrate is an animal. Each animal is not unicellular. Arthropods are invertebrates. Insects are six-legged. Every arthropod is segmented. Every butterfly is a lepidopteran. Alex is a butterfly. True or false: Alex is six-legged.\nA:',
        'answer': 'Alex is a butterfly.\nEvery butterfly is a lepidopteran. So Alex is a lepidopteran.\nEvery lepidopteran is an insect. So Alex is an insect.\nInsects are six-legged. So Alex is six-legged.\nThe answer is true.'
    },
    {
        'question': 'Q: Natural numbers are not negative. Integers are real numbers. Every real number is a number. Each prime number is prime. Mersenne primes are prime. Natural numbers are integers. Each Mersenne prime is a prime number. Negative numbers are negative. Each real number is real. Prime numbers are natural numbers. 31 is a Mersenne prime. True or false: 31 is not negative.\nA:',
        'answer': '31 is a Mersenne prime.\nEach Mersenne prime is a prime number. So 31 is a prime number.\nPrime numbers are natural numbers. So 31 is a natural number.\nNatural numbers are not negative. So 31 is not negative.\nThe answer is true.'
    },
    {
        'question': 'Q: Arthropods are not bony. Butterflies are lepidopterans. Every insect is an arthropod. Each invertebrate is an animal. Each lepidopteran is an insect. Each insect is not eight-legged. Every animal is multicellular. Spiders are eight-legged. Arthropods are invertebrates. Fae is a butterfly. True or false: Fae is not eight-legged.\nA:',
        'answer': 'Fae is a butterfly.\nButterflies are lepidopterans. So Fae is a lepidopteran.\nEach lepidopteran is an insect. So Fae is an insect.\nEach insect is not eight-legged. So Fae is not eight-legged.\nThe answer is true.'
    },
    {
        'question': 'Q: Every invertebrate is an animal. Arthropods are segmented. Every arthropod is an invertebrate. Nematodes are not segmented. Animals are not unicellular. Every insect is not eight-legged. Insects are arthropods. Every lepidopteran is an insect. Butterflies are lepidopterans. Alex is a butterfly. True or false: Alex is not segmented.\nA:',
        'answer': 'Alex is a butterfly.\nButterflies are lepidopterans. So Alex is a lepidopteran.\nEvery lepidopteran is an insect. So Alex is an insect.\nInsects are arthropods. So Alex is an arthropod.\nArthropods are segmented. So Alex is segmented.\nThe answer is false.'
    },
    {
        'question': 'Q: Lepidopterans are insects. Each butterfly is a lepidopteran. Insects are not eight-legged. Arthropods are protostomes. Every protostome is an invertebrate. Insects are arthropods. Every invertebrate is an animal. Every nematode is not segmented. Painted ladies are butterflies. Animals are not unicellular. Every arthropod is segmented. Sally is a painted lady. True or false: Sally is segmented.\nA:',
        'answer': 'Sally is a painted lady.\nPainted ladies are butterflies. So Sally is a butterfly.\nEach butterfly is a lepidopteran. So Sally is a lepidopteran.\nLepidopterans are insects. So Sally is an insect.\nInsects are arthropods. So Sally is an arthropod.\nEvery arthropod is segmented. So Sally is segmented.\nThe answer is true.'
    },
    {
        'question': 'Q: Every real number is a number. Each Mersenne prime is a prime number. Each integer is a real number. Every real number is real. Every prime number is a natural number. Imaginary numbers are not real. Every prime number is not composite. Each Mersenne prime is not composite. Natural numbers are not negative. Each natural number is an integer. 127 is a prime number. True or false: 127 is real.\nA:',
        'answer': '127 is a prime number.\nEvery prime number is a natural number. So 127 is a natural number.\nEach natural number is an integer. So 127 is an integer.\nEach integer is a real number. So 127 is a real number.\nEvery real number is real. So 127 is real.\nThe answer is true.'
    },
    {
        'question': 'Q: Imaginary numbers are not real. Prime numbers are natural numbers. Real numbers are real. Natural numbers are positive. Each real number is a number. Each prime number is not composite. Each Mersenne prime is a prime number. Each Mersenne prime is not composite. Natural numbers are integers. Each integer is a real number. 131071 is a prime number. True or false: 131071 is not real.\nA:',
        'answer': '131071 is a prime number.\nPrime numbers are natural numbers. So 131071 is a natural number.\nNatural numbers are integers. So 131071 is an integer.\nEach integer is a real number. So 131071 is a real number.\nReal numbers are real. So 131071 is real.\nThe answer is false.'
    },
    {
        'question': 'Q: Natural numbers are integers. Each real number is not imaginary. Negative numbers are negative. Every integer is a real number. Every natural number is not negative. Every prime number is prime. Real numbers are numbers. Mersenne primes are prime. Mersenne primes are prime numbers. Every prime number is a natural number. 31 is a Mersenne prime. True or false: 31 is negative.\nA:',
        'answer': '31 is a Mersenne prime.\nMersenne primes are prime numbers. So 31 is a prime number.\nEvery prime number is a natural number. So 31 is a natural number.\nEvery natural number is not negative. So 31 is not negative.\nThe answer is false.'
    }
]

PROMPT_TEMPLATE = "Q: {question} {query}\nA:"

ACTION_STEP_SEPARATOR = "\n"

MAX_STEP_TOKENS = 64

def extract_answer(text: str) -> str:
    steps = [s.split("So")[1].strip()+'.' for s in text.split('.') if "So" in s]
        
    return "\n".join(steps)

def check_answer_correctness(ground_truth: str, response: str):
    try:
        if ground_truth == response:
            return True
    except Exception:
        pass
    
    return False

class ProntoAction(Action):
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
    

class ProntoState(State):
    problem: list[dict]
    problem_ids: Optional[list[int]] = None
    answer: str
    trace: list[ProntoAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "\n\n".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, ProntoState) and str(self) == str(other)
    
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
        return bool(re.search(r"\b[Tt]he answer is\b", self.trace[-1].text)) or self.trace[-1].is_final_action()
 
 
class ProntoQA(Task):
    data: Dataset
    num_shot: int
    model: LLM
    inference_config: InferenceConfig
    reward_coeff: dict[str, float]
    answer_judge: Optional[BaseJudge]
    step_judge: Optional[BaseJudge]
    
    @classmethod
    def from_config(cls, model: LLM, inference_config: InferenceConfig, task_config: OmegaConf):
        # test data
        data = load_dataset(
            "json", 
            data_files={
                "train": os.path.join(task_config.data_dir, "train.json"),
                "test": os.path.join(task_config.data_dir, "test.json")
            }
        )[task_config.split]
        if task_config.max_num_instances > 0:
            data = data.select(range(min(task_config.max_num_instances, len(data))))
            
        answer_judge = None
        step_judge = None
        
        return cls(
            data=data,
            num_shot=task_config.num_shot,
            model=model,
            inference_config=inference_config,
            reward_coeff=task_config.reward_coeff,
            answer_judge=answer_judge,
            step_judge=step_judge,
        )
        
    @property
    def size(self):
        return len(self.data)
    
    def init(self, index) -> ProntoState:
        messages = []
        # select ICL examples
        for icl_example in IN_CONTEXT_EXAMPLES[:self.num_shot]:
            messages.append({"role": "user", "content": icl_example["question"]})
            messages.append({"role": "assistant", "content": icl_example["answer"]})
        
        # select test examples
        test_example = self.data[index]
        test_prompt = PROMPT_TEMPLATE.format(question=test_example["question"], query=test_example["query"])
        messages.append({"role": "user", "content": test_prompt})
        
        # encode problem
        problem_ids = self.model.encode(messages, self.inference_config)
        
        # create init state
        init_state = ProntoState(
            problem=messages,
            problem_ids=problem_ids,
            answer="\n".join(test_example["chain_of_thought"][2::2]),
            trace=[],
            embedding=self.model.get_prompt_embedding(messages=messages, inference_config=InferenceConfig())
        )
        
        return init_state
    
    def transition(self, state: ProntoState, action: ProntoAction) -> ProntoState:
        return ProntoState(
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding
        )
        
    async def step(self, state: ProntoState, action: ProntoAction) -> tuple[ProntoState, float, bool, dict]:
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
    
    def propose_actions(self, state: ProntoState, num_actions: int) -> list[ProntoAction]:
        inference_config = self.inference_config.model_copy(update={"model_config": ConfigDict(frozen=False)})
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
            ProntoAction(
                text=response.text,
                response_ids=response.token_ids[len(state.problem_ids):] if response.token_ids is not None else None,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: ProntoState):
        problem = state.problem[-1]["content"]
        response = state.to_response()
        
        score = self.answer_judge.judge(QUESTION=problem, ANSWER=response)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_action(self, state: ProntoState, action: ProntoAction):
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