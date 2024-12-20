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
        "question": "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:",
        "answer": "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n",
    },
    {
        "question": "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:",
        "answer": "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n",
    },
    {
        "question": "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:",
        "answer": "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n",
    },
    {
        "question": "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:",
        "answer": "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n"
    }
]

PROMPT_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"

ACTION_STEP_SEPARATOR = "\n"

MAX_STEP_TOKENS = 64

# adapted from autorace
ANSWER_SELFEVAL_SYSTEM_PROMPT = """You are a teacher who evaluates students' answers based on specific criteria. You will be provided with a question and a student's answer with step-by-step reasoning. You are required to check the correctness of the reasoning chains step by step. The criterions are as follows:

**Accuracy in Mathematical Operations:** Ensure calculations are correct and follow logical mathematical principles.

**Understanding the Problem Statement:** Comprehend the details and conditions of the question accurately.

**Correct Application of Mathematical Concepts:** Apply the right mathematical formulas, operations, or concepts to solve the problem.

**Unit Conversion and Appropriateness:** When required, correctly convert units and use appropriate units in the answer.

**Final Answer Relevance:** Ensure the final answer directly addresses the question asked, and is presented clearly and concisely.

**Logical Reasoning and Step-by-Step Explanation:** The answer should include a logical, step-by-step explanation that demonstrates how the final answer was reached."""

ANSWER_SELFEVAL_USER_PROMPT = """Below is a question and an answer from a student:

Question: {QUESTION}

Student answer: {ANSWER}

Please check the answer through each criterion, and make sure you carefully examine each reasoning step. Finally, if there is any step that fails the verification, output 'The answer is incorrect', otherwise output 'The answer is correct'."""


STEP_SELFEVAL_SYSTEM_PROMPT = """You are a teacher who evaluates students' answers based on specific criteria. You will be provided with a question, the student's previous reasoning steps, and the current reasoning step. Your task is to assess the current reasoning step based on the following criteria:

**Accuracy in Mathematical Operations:** Ensure calculations are correct and follow logical mathematical principles.

**Understanding the Problem Statement:** Comprehend the details and conditions of the question accurately.

**Correct Application of Mathematical Concepts:** Apply the right mathematical formulas, operations, or concepts to solve the problem.

**Unit Conversion and Appropriateness:** When required, correctly convert units and use appropriate units in the answer.

**Logical Reasoning:** The reasoning step should follow a logical flow from the previous steps and the problem statement."""

STEP_SELFEVAL_USER_PROMPT = """Below is a question and an answer from a student:

Question: {QUESTION}

Previous reasoning steps: {PREVIOUS_STEPS}

Current reasoning step: {CURRENT_STEP}

Please evaluate the current reasoning step based on the criteria mentioned above. If the step satisfies all the criteria, output 'The step is correct'. If any criterion is not met, output 'The step is incorrect'."""


# These postprocessing functions are adapted from opencompass
def extract_answer(text: str):
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    
    return numbers[-1]


def check_answer_correctness(ground_truth: str, response: str):
    try:
        if ground_truth == response or abs(float(response) - int(ground_truth)) < 1e-6:
            return True
    except Exception:
        pass
    
    return False


class GSM8KAction(Action):
    text: str
    finish_reason: str
    log_prob: Optional[float] = None
    confidence: Optional[float] = None
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return self.text
    
    def is_final_action(self):
        return self.finish_reason == "stop"
    
    
class GSM8KState(State):
    problem: list[dict]
    answer: str
    trace: list[GSM8KAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "\n\n".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, GSM8KState) and str(self) == str(other)
    
    def to_messages(self):
        messages = self.problem.copy() # copy to avoid changing the problem
        if self.trace:
            assistant_prefill = self.to_response() + ACTION_STEP_SEPARATOR
            messages.append({"role": "assistant", "content": assistant_prefill})
            
        return messages
    
    def to_response(self):
        return ACTION_STEP_SEPARATOR.join([action.text for action in self.trace]).lstrip()
    
    def is_terminal(self):
        if not self.trace:
            return False
        return bool(re.search(r"\b[Tt]he answer is\b", self.trace[-1].text)) or self.trace[-1].is_final_action()


class GSM8K(Task):
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
        data = load_dataset("gsm8k", "main", split=task_config.split)
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
        
    def init(self, index) -> GSM8KState:
        messages = []
        # select ICL examples
        for icl_example in IN_CONTEXT_EXAMPLES[:self.num_shot]:
            messages.append({"role": "user", "content": icl_example["question"]})
            messages.append({"role": "assistant", "content": icl_example["answer"]})
        
        # select test examples
        test_example = self.data[index]
        test_prompt = PROMPT_TEMPLATE.format(question=test_example["question"])
        messages.append({"role": "user", "content": test_prompt})
        
        # create init state
        init_state = GSM8KState(
            problem=messages,
            answer=test_example["answer"],
            trace=[],
            embedding=self.model.get_prompt_embedding(messages=messages, inference_config=InferenceConfig())
        )
        
        return init_state
    
    def transition(self, state: GSM8KState, action: GSM8KAction) -> GSM8KState:
        return GSM8KState(
            problem=state.problem,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding
        )
    
    async def step(self, state: GSM8KState, action: GSM8KAction) -> tuple[GSM8KState, float, bool, dict]:
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
    
    def propose_actions(self, state: GSM8KState, num_actions: int) -> list[GSM8KAction]:
        messages = state.to_messages()
        inferece_config = self.inference_config.model_copy(update={"model_config": ConfigDict(frozen=False)})
        inferece_config.stop_sequences += [ACTION_STEP_SEPARATOR]
        inferece_config.max_tokens = MAX_STEP_TOKENS
        llm_responses: list[LLMResponse] = self.model.batch_call(
            batch_messages=[messages] * num_actions,
            inference_config=inferece_config,
        )
        llm_responses = list(dict.fromkeys(llm_responses)) # remove duplicated responses
        
        return [
            GSM8KAction(
                text=response.text,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: GSM8KState):
        problem = state.problem[-1]["content"]
        response = state.to_response()
        
        score = self.answer_judge.judge(QUESTION=problem, ANSWER=response)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_action(self, state: GSM8KState, action: GSM8KAction):
        problem = state.problem[-1]["content"]
        previous_response = state.to_response()
        current_step = action.text
        
        score = self.step_judge.judge(QUESTION=problem, PREVIOUS_STEPS=previous_response, CURRENT_STEP=current_step)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_solution(self, answer: str, solutions: list[Solution]):
        ground_truth = answer.split('#### ')[1].replace(',', '')
        
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