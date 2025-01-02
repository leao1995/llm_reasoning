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

# adapted from llm-reasoner
IN_CONTEXT_EXAMPLES = [
    {
        "question": "Question: A class of 35 students has an average height of 180 cm. Seven students whose average height is 120 cm, left the class and seven others whose average height is 140 cm, joined. Calculate the new average height of the students of the class (in cm) is?\nOptions: A)204.6 cm, B)404.6 cm, C)224.6 cm, D)184.0 cm, E)256.6 cm\nAnswer:",
        "answer": "The total height of students before seven students left is 180 * 35 = 6300 cm.\nThe total height of students who joined is 140 * 7  = 980 cm.\nThe new total height of students after seven students joined is 6300 - 840 + 980 = 6440 cm.\nThe new average height is 6440 / 35 = 184 cm.\nThe answer is D."
    },
    {
        "question": "Question: How much is 70% of 40 is greater than 4/5 of 25?\nOptions: A)22, B)67, C)88, D)12, E)8\nAnswer:",
        "answer": "70% of 40 is 40 * 0.7 = 28.\n4/5 of 25 is 25 * 4/5 = 20.\n70% of 40 is greater than 4/5 of 25 by 28 - 20 = 8.\nThe answer is E."
    },
    {
        "question": "Question: What is the average of the first 21 multiples of 7?\nOptions: A)22, B)77, C)88, D)21, E)65\nAnswer:",
        "answer": "The sum of the first 21 multiples of 7 is 7 * (1+2+….+21).\nAfter simplification, 7 * ((21x22) / 2) = 1617.\nThe average of the first 21 multiples of 7 is 1617 / 21 = 77.\nThe answer is B."
    },
    {
        "question": "Question: In a certain store, the profit is 320% of the cost. If the cost increases by 25% but the selling price remains constant, approximately what percentage of the selling price is the profit?\nOptions: A)80%, B)70%, C)60%, D)50%, E)40%\nAnswer:",
        "answer": "Let cost be 100.\nThe profit is 100 * 3.2 = 320.\nThe selling price is 100 + 320 = 420.\nThe new cost is 100 * 0.25 + 100 = 125.\nThe new profit is 420 - 125 = 295.\nThe percentage is 296/420 * 100 = 70%.\nThe answer is B."
    },
    {
        "question": "Question: A 290 meters long train running at the speed of 120 kmph crosses another train running in opposite direction at the speed of 80 kmph in 9 seconds. What is the length of the other train?\nOptions: A)230 m, B)210 m, C)260 m, D)320 m, E)330 m\nAnswer:",
        "answer": "The relative speed is 120 + 80 = 200 kmph.\nThe speed in m/s is 200 kmph = (200 * 1000)/3600m/s = 55.55m/s.\nThe distance is 55.55m/s * 9 = 499.95m.\nThe length of the other train is 499.95 - 290 = 209.95.\nThe answer is B."
    },
    {
        "question": "Question: A train, 140 meters long, travels at a speed of 45 km/hr and crosses a bridge in 30 seconds. The length of the bridge is\nOptions: A)270 m, B)245 m, C)235 m, D)220 m, E)240 m\nAnswer:",
        "answer": "The speed in m/s is 45 km/hr = (45 * 1000)/3600 m/s = 12.5 m/s.\nThe total distance traveled is 12.5 m/s * 30 s = 375 m.\nThe length of the bridge is 375 m - 140 m = 235 m.\nThe answer is C."
    },
    {
        "question": "Question: A coin is tossed three times. What is the probability that there is at the least one tail?\nOptions: A)7/8, B)6/7, C)1/7, D)3/7, E)1/31\nAnswer:",
        "answer": "The probability of getting all heads is (1/2)^3 = 1/8.\nThe probability of at least one tail is 1 - 1/8 = 7/8.\nThe answer is A."
    },
    {
        "question": "Question: 10 men, working 3 hours a day can complete a work in 18 days. How many hours a day must 15 men work to complete the work in 12 days?\nOptions: A)3 hours a day, B)5 hours a day, C)6 hours a day, D)7 hours a day, E)8 hours a day\nAnswer:",
        "answer": "The total man-hours needed to complete the work is 10 * 3 * 18 = 540.\nThe hours needed to complete the work in 12 days are 540 / (15 * 12) = 540 / 180 = 3.\nThe answer is A."
    },
    {
        "question": "Question: When 20 percent of a number is added to another number the second number increases to 140 per cent. What is the ratio between the first and the second number?\nOptions: A)3 : 4, B)2 : 1, C)3 : 2, D)Data inadequate, E)None of these\nAnswer:",
        "answer": "Let the first number be x and the second number be y.\nThe equation satisfied the increase is y + 0.2 x = 1.4y.\nThe ratio between the first and the second number is y = 0.5x.\nThe answer is B."
    },
    {
        "question": "Question: An electric pump can fill a tank in 7 hours. Because of a leak in the tank, it took 14 hours to fill the tank. If the tank is full, how much time will the leak take to empty it?\nOptions: A)10 hours, B)12 hours, C)8 hours, D)5 hours, E)14 hours\nAnswer:",
        "answer": "The filling rate of the pump without leak is 1/7.\nThe filling rate of the pump with leak is 1/14.\nThe leaking rate is 1/7 - 1/14 = 1/14.\nSo it takes 14 hours to empty it.\nThe answer is E."
    }
]

PROMPT_TEMPLATE = "Question: {question}\nOptions: {options}\nAnswer:"

ACTION_STEP_SEPARATOR = "\n"

MAX_STEP_TOKENS = 64

# adapted from autorace
ANSWER_SELFEVAL_SYSTEM_PROMPT = """You are a teacher who evaluates students' answers based on specific criteria. You will be provided with a question and a student's answer with step-by-step reasoning. You are required to check the correctness of the reasoning chains step by step. The criterions are as follows:

**Accuracy of Mathematical Calculation: Ensure the student's calculations are mathematically correct and follow logical steps.

**Relevance and Application of Concepts: Check if the student has applied relevant mathematical concepts and formulas appropriately to solve the problem.

**Understanding of the Problem: Evaluate whether the student demonstrates a clear understanding of the problem's requirements and conditions.

**Logical and Coherent Solution: The student's solution should be logical, coherent, and follow a clear, step-by-step process.

**Final Answer Alignment: The student's final answer should align with the options provided (if applicable) and correctly solve the problem as stated.

**Misinterpretation or Misapplication: Note any instances where the student misinterprets the problem or misapplies mathematical concepts, leading to an incorrect solution."""

ANSWER_SELFEVAL_USER_PROMPT = """Below is a question and an answer from a student:

Question: {QUESTION}

Student answer: {ANSWER}

Please check the answer through each criterion, and make sure you carefully examine each reasoning step. Finally, if there is any step that fails the verification, output 'The answer is incorrect', otherwise output 'The answer is correct'."""

STEP_SELFEVAL_SYSTEM_PROMPT = """You are a teacher who evaluates students' answers based on specific criteria. You will be provided with a question, the student's previous reasoning steps, and the current reasoning step. Your task is to assess the current reasoning step based on the following criteria:

**Accuracy of Mathematical Calculation: Ensure the student's calculations in the current step are mathematically correct and follow logical steps.

**Relevance and Application of Concepts: Check if the student has applied relevant mathematical concepts and formulas appropriately in the current step to progress towards solving the problem.

**Logical Progression: Evaluate whether the current step logically follows from the previous steps and contributes towards solving the problem.

**Understanding of the Problem: Assess if the student demonstrates a clear understanding of the problem's requirements and conditions in the current step.

**Misinterpretation or Misapplication: Note any instances where the student misinterprets the problem or misapplies mathematical concepts in the current step, potentially leading to an incorrect solution."""

STEP_SELFEVAL_USER_PROMPT = """Below is a question and an answer from a student:

Question: {QUESTION}

Previous reasoning steps: {PREVIOUS_STEPS}

Current reasoning step: {CURRENT_STEP}

Please evaluate the current reasoning step based on the criteria mentioned above. If the step satisfies all the criteria, output 'The step is correct'. If any criterion is not met, output 'The step is incorrect'."""

# These postprocessing functions are adapted from llm-reasoner
def extract_answer(text: str) -> str:
    match = re.search(r'.*[Tt]he answer is.*?([A-E]).*?$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''


def check_answer_correctness(ground_truth: str, response: str):
    try:
        if ground_truth == response:
            return True
    except Exception:
        pass
    
    return False


class AquaAction(Action):
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

    
class AquaState(State):
    problem: list[dict]
    problem_ids: Optional[list[int]] = None
    answer: str
    trace: list[AquaAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "\n\n".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, AquaState) and str(self) == str(other)
    
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


class Aqua(Task):
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
        data = load_dataset("deepmind/aqua_rat", "raw", split=task_config.split)
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
    
    def init(self, index) -> AquaState:
        messages = []
        # select ICL examples
        for icl_example in IN_CONTEXT_EXAMPLES[:self.num_shot]:
            messages.append({"role": "user", "content": icl_example["question"]})
            messages.append({"role": "assistant", "content": icl_example["answer"]})
        
        # select test examples
        test_example = self.data[index]
        test_prompt = PROMPT_TEMPLATE.format(question=test_example["question"], options=", ".join(test_example["options"]))
        messages.append({"role": "user", "content": test_prompt})
        
        # encode problem
        problem_ids = self.model.encode(messages, self.inference_config)
        
        # create init state
        init_state = AquaState(
            problem=messages,
            problem_ids=problem_ids,
            answer=test_example["correct"],
            trace=[],
            embedding=self.model.get_prompt_embedding(messages=messages, inference_config=InferenceConfig())
        )
        
        return init_state
    
    def transition(self, state: AquaState, action: AquaAction) -> AquaState:
        return AquaState(
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding
        )
        
    async def step(self, state: AquaState, action: AquaAction) -> tuple[AquaState, float, bool, dict]:
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
    
    def propose_actions(self, state: AquaState, num_actions: int) -> list[AquaAction]:
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
            AquaAction(
                text=response.text,
                response_ids=response.token_ids[len(state.problem_ids):] if response.token_ids is not None else None,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: AquaState):
        problem = state.problem[-1]["content"]
        response = state.to_response()
        
        score = self.answer_judge.judge(QUESTION=problem, ANSWER=response)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_action(self, state: AquaState, action: AquaAction):
        problem = state.problem[-1]["content"]
        previous_response = state.to_response()
        current_step = action.text
        
        score = self.step_judge.judge(QUESTION=problem, PREVIOUS_STEPS=previous_response, CURRENT_STEP=current_step)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_solution(self, answer: str, solutions: list[Solution]):
        ground_truth = answer.strip()
        
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