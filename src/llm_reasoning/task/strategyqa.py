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
IN_CONTEXT_EXAMPLES = [
    {
        "question": "Question: Do hamsters provide food for any animals?\nAnswer:",
        "answer": "Hamsters are prey animals.\nPrey are food for predators.\nThus, hamsters provide food for some animals.\nSo the answer is yes."
    },
    {
        "question": "Question: Could Brooke Shields succeed at University of Pennsylvania?\nAnswer:",
        "answer": "Brooke Shields went to Princeton University.\nPrinceton University is about as academically rigorous as the University of Pennsylvania.\nThus, Brooke Shields could also succeed at the University of Pennsylvania.\nSo the answer is yes."
    },
    {
        "question": "Question: Hydrogen\'s atomic number squared exceeds number of Spice Girls?\nAnswer:",
        "answer": "Hydrogen has an atomic number of 1.\n1 squared is 1.\nThere are 5 Spice Girls.\nThus, Hydrogen\'s atomic number squared is less than 5.\nSo the answer is no."
    },
    {
        "question": "Question: Is it common to see frost during some college commencements?\nAnswer:",
        "answer": "College commencement ceremonies can happen in December, May, and June.\nDecember is in the winter, so there can be frost.\nThus, there could be frost at some commencements.\nSo the answer is yes."
    },
    {
        "question": "Question: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nAnswer:",
        "answer": "The War in Vietnam was 6 months.\nThe gestation period for a llama is 11 months, which is more than 6 months.\nThus, a llama could not give birth twice during the War in Vietnam.\nSo the answer is no."
    },
    {
        "question": "Question: Would a pear sink in water?\nAnswer:",
        "answer": "The density of a pear is about 0.6g/cm3, which is less than water.\nObjects less dense than water float.\nThus, a pear would float.\nSo the answer is no."
    }
]

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

ACTION_STEP_SEPARATOR = "\n"

MAX_STEP_TOKENS = 64

# adapted from autorace
ANSWER_SELFEVAL_SYSTEM_PROMPT = """You are a teacher who evaluates students' answers based on specific criteria. You will be provided with a question and a student's answer with step-by-step reasoning. You are required to check the correctness of the reasoning chains step by step. The criterions are as follows:

**Accuracy: The answer must correctly address the question and align with factual information.

**Relevance: The answer should directly address the specific question asked without deviating onto unrelated topics.

**Evidence and Logic: The answer should be supported by logical reasoning and, when applicable, evidence or examples.

**Clarity and Coherence: The answer should be clearly articulated and logically structured.

**Adherence to the Question's Context: The answer must respect the context or scenario presented in the question.

**Avoidance of Assumptions: The answer should not include unfounded assumptions or speculation not supported by the question or general knowledge."""

ANSWER_SELFEVAL_USER_PROMPT = """Below is a question and an answer from a student:

Question: {QUESTION}

Student answer: {ANSWER}

Please check the answer through each criterion, and make sure you carefully examine each reasoning step. Finally, if there is any step that fails the verification, output 'The answer is incorrect', otherwise output 'The answer is correct'."""

STEP_SELFEVAL_SYSTEM_PROMPT = """You are a teacher who evaluates student's answers based on specific criteria. You will be provided with a question, the student's previous reasoning steps, and the current step to evaluate. Your task is to check if the current step is correct and logically follows from the previous steps, given the context of the question. The criterions are as follows:

**Logical Coherence: The current step should logically follow from the previous steps and the question context, without introducing any logical gaps or inconsistencies.

**Relevance: The current step should directly contribute to answering the question and should not deviate onto unrelated topics.

**Evidence and Logic: The current step should be supported by logical reasoning and, when applicable, evidence or examples from the previous steps or the question context.

**Clarity: The current step should be clearly articulated and easily understandable.

**Adherence to the Context: The current step must respect the context or scenario presented in the question and should not introduce any contradictions that violate the given context.

**Avoidance of Assumptions: The current step should not include unfounded assumptions or speculation not supported by the question, previous steps, or general knowledge.

**Factual Accuracy: The current step should not contain any factual errors or inaccuracies that contradict established knowledge or information provided in the question."""

STEP_SELFEVAL_USER_PROMPT = """Below is a question and an answer from a student:

Question: {QUESTION}

Previous reasoning steps: {PREVIOUS_STEPS}

Current reasoning step: {CURRENT_STEP}

Please evaluate the current reasoning step based on the criteria mentioned above. If the step satisfies all the criteria, output 'The step is correct'. If any criterion is not met, output 'The step is incorrect'."""


# These postprocessing functions are adapted from opencompass
def extract_answer(text: str) -> str:
    text = text.split('answer is ')[-1]
    match = re.search(r'(yes|no)', text.lower())
    if match:
        return match.group(1)
    return ''


def check_answer_correctness(ground_truth: str, response: str):
    try:
        if ground_truth == response:
            return True
    except Exception:
        pass
    
    return False


class SQAction(Action):
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
    

class SQState(State):
    problem: list[dict]
    problem_ids: Optional[list[int]] = None
    answer: bool
    trace: list[SQAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "\n\n".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, SQState) and str(self) == str(other)
    
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
    

class StrategyQA(Task):
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
        raw_data = load_dataset("wics/strategy-qa", split="test")
        datasets = raw_data.train_test_split(test_size=1000, shuffle=True, seed=42)
        data = datasets[task_config.split]
        if task_config.max_num_instances > 0:
            data = data.select(range(min(task_config.max_num_instances, len(data))))
        
        # judge for intermediate steps and final answer
        if task_config.judge_type == "llm":
            answer_judge = LLMJudge(
                model=model,
                inference_config=inference_config,
                system_prompt=ANSWER_SELFEVAL_SYSTEM_PROMPT,
                system_prompt_vars=[],
                user_prompt=ANSWER_SELFEVAL_USER_PROMPT,
                user_prompt_vars=["QUESTION", "ANSWER"],
                post_processor=LikertScaleProcessor(scales={"The answer is incorrect": 0, "The answer is correct": 1})
            )
            
            step_judge = LLMJudge(
                model=model,
                inference_config=inference_config,
                system_prompt=STEP_SELFEVAL_SYSTEM_PROMPT,
                system_prompt_vars=[],
                user_prompt=STEP_SELFEVAL_USER_PROMPT,
                user_prompt_vars=["QUESTION", "PREVIOUS_STEPS", "CURRENT_STEP"],
                post_processor=LikertScaleProcessor(scales={"The step is incorrect": 0, "The step is correct": 1})
            )
        elif task_config.judge_type == "likelihood":
            answer_judge = LikelihoodJudge(
                model=model,
                inference_config=inference_config,
                system_prompt=ANSWER_SELFEVAL_SYSTEM_PROMPT,
                system_prompt_vars=[],
                user_prompt=ANSWER_SELFEVAL_USER_PROMPT,
                user_prompt_vars=["QUESTION", "ANSWER"],
                candidates=["The answer is incorrect", "The answer is correct"],
                score_idx=1
            )
            
            step_judge = LikelihoodJudge(
                model=model,
                inference_config=inference_config,
                system_prompt=STEP_SELFEVAL_SYSTEM_PROMPT,
                system_prompt_vars=[],
                user_prompt=STEP_SELFEVAL_USER_PROMPT,
                user_prompt_vars=["QUESTION", "PREVIOUS_STEPS", "CURRENT_STEP"],
                candidates=["The step is incorrect", "The step is correct"],
                score_idx=1
            )
        else:
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
    
    def init(self, index) -> SQState:
        messages = []
        # select ICL examples
        for icl_example in IN_CONTEXT_EXAMPLES[:self.num_shot]:
            messages.append({"role": "user", "content": icl_example["question"]})
            messages.append({"role": "assistant", "content": icl_example["answer"]})
        
        # select test examples
        test_example = self.data[index]
        test_prompt = PROMPT_TEMPLATE.format(question=test_example["question"])
        messages.append({"role": "user", "content": test_prompt})
        
        # encode problem
        problem_ids = self.model.encode(messages, self.inference_config)
        
        # create init state
        init_state = SQState(
            problem=messages,
            problem_ids=problem_ids,
            answer=test_example["answer"],
            trace=[],
            embedding=self.model.get_prompt_embedding(messages=messages, inference_config=InferenceConfig())
        )
        
        return init_state
    
    def transition(self, state: SQState, action: SQAction) -> SQState:
        return SQState(
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding
        )
        
    async def step(self, state: SQState, action: SQAction) -> tuple[SQState, float, bool, dict]:
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
    
    def propose_actions(self, state: SQState, num_actions: int) -> list[SQAction]:
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
            SQAction(
                text=response.text,
                response_ids=response.token_ids[len(state.problem_ids):] if response.token_ids is not None else None,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: SQState):
        problem = state.problem[-1]["content"]
        response = state.to_response()
        
        score = self.answer_judge.judge(QUESTION=problem, ANSWER=response)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_action(self, state: SQState, action: SQAction):
        problem = state.problem[-1]["content"]
        previous_response = state.to_response()
        current_step = action.text
        
        score = self.step_judge.judge(QUESTION=problem, PREVIOUS_STEPS=previous_response, CURRENT_STEP=current_step)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_solution(self, answer: bool, solutions: list[Solution]):
        ground_truth = 'yes' if answer else 'no'
        
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