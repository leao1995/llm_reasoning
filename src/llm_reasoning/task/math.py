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
        "question": "Find the domain of the expression $\\frac{{\sqrt{{x-2}}}}{{\sqrt{{5-x}}}}$.}}\nPlease reason step by step, and put your final answer within \\boxed{{}}",
        "answer": "The expressions inside each square root must be non-negative.\nTherefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$.\nAlso, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.\nTherefore, the domain of the expression is $\\boxed{{[2,5)}}$."
    },
    {
        "question": "If $\det \mathbf{{A}} = 2$ and $\det \mathbf{{B}} = 12,$ then find $\det (\mathbf{{A}} \mathbf{{B}}).$\nPlease reason step by step, and put your final answer within \\boxed{{}}",
        "answer": "We have that $\det (\mathbf{{A}} \mathbf{{B}}) = (\det \mathbf{{A}})(\det \mathbf{{B}}) = (2)(12) = \\boxed{{24}}."
    },
    {
        "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\nPlease reason step by step, and put your final answer within \\boxed{{}}",
        "answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.\nIf he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.\nEquating this to 480 pounds, we can solve for $n$: \\begin{{align*}} 30n&=480\\\\ \Rightarrow\qquad n&=480/30=\\boxed{{16}} \end{{align*}}"
    },
    {
        "question": "If the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.\nPlease reason step by step, and put your final answer within \\boxed{{}}",
        "answer": "If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain $$6y-9x=-\\frac{{3}}{{2}}a.$$\nSince we also know that $6y-9x=b$, we have $$-\\frac{{3}}{{2}}a=b\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$"
    }
]

PROMPT_TEMPLATE = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

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
def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def extract_answer(text: str):
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    for maybe_ans in text.split('.'):
        # if 'final answer' in maybe_ans.lower():
        if re.search('final answer|answer is', maybe_ans.lower()):
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split('.')[0])


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string


def _fix_sqrt(string):
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string
    
    
def _strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace('\n', '')

    # right "."
    string = string.rstrip('.')

    # remove inverse spaces
    string = string.replace('\\!', '')
    string = string.replace('\\ ', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        string = _string

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')
    string = string.replace('$', '')

    string = string.replace('\\text', '')
    string = string.replace('x\\in', '')

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605
    string = string.replace('%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
    # add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    # cdot
    string = string.replace('\\cdot', '')

    # inf
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')

    # and
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')

    # use regex to remove \mbox{...}
    string = re.sub(r'\\mbox{.*?}', '', string)

    # quote
    string.replace("'", '')
    string.replace('"', '')

    # i, j
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0+$', r'\1', string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = _fix_sqrt(string)
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
    # cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def check_answer_correctness(str1, str2):
    if str1 is None and str2 is None:
        print('WARNING: Both None')
        return True
    if str1 is None or str2 is None:
        return False
    
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if ss1 == ss2:
            return True
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass
    
    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    return str1 == str2


class MathAction(Action):
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


class MathState(State):
    problem: list[dict]
    problem_ids: Optional[list[int]] = None
    answer: str
    trace: list[MathAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "\n\n".join(msg["content"] for msg in self.to_messages())
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return isinstance(other, MathState) and str(self) == str(other)
    
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
        return bool(re.search(r"\\boxed\{.*?\}", self.trace[-1].text)) or self.trace[-1].is_final_action()


class Math(Task):
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
        if task_config.split == "train":
            data = load_dataset("hendrycks/competition_math", trust_remote_code=True, split="train")
            data = data.shuffle(seed=42)
        else:
            data = load_dataset("HuggingFaceH4/MATH-500", split="test")
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
    
    def init(self, index) -> MathState:
        messages = []
        # select ICL examples
        for icl_example in IN_CONTEXT_EXAMPLES[:self.num_shot]:
            messages.append({"role": "user", "content": icl_example["question"]})
            messages.append({"role": "assistant", "content": icl_example["answer"]})
        
        # select test examples
        test_example = self.data[index]
        test_prompt = PROMPT_TEMPLATE.format(problem=test_example["problem"])
        messages.append({"role": "user", "content": test_prompt})
        
        # encode problem
        problem_ids = self.model.encode(messages, self.inference_config)
        
        # create init state
        init_state = MathState(
            problem=messages,
            problem_ids=problem_ids,
            answer=test_example["solution"],
            trace=[],
            embedding=self.model.get_prompt_embedding(messages=messages, inference_config=InferenceConfig())
        )
        
        return init_state
    
    def transition(self, state: MathState, action: MathAction) -> MathState:
        return MathState(
            problem=state.problem,
            problem_ids=state.problem_ids,
            answer=state.answer,
            trace=state.trace + [action],
            embedding=action.embedding
        )
        
    async def step(self, state: MathState, action: MathAction) -> tuple[MathState, float, bool, dict]:
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
    
    def propose_actions(self, state: MathState, num_actions: int) -> list[MathAction]:
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
            MathAction(
                text=response.text,
                response_ids=response.token_ids[len(state.problem_ids):] if response.token_ids is not None else None,
                finish_reason=response.finish_reason,
                log_prob=sum(response.logprobs) if response.logprobs else None,
                confidence=mean(response.confidences) if response.confidences else None,
                embedding=response.embedding,
            )
            for response in llm_responses
        ]
        
    def eval_state(self, state: MathState):
        problem = state.problem[-1]["content"]
        response = state.to_response()
        
        score = self.answer_judge.judge(QUESTION=problem, ANSWER=response)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_action(self, state: MathState, action: MathAction):
        problem = state.problem[-1]["content"]
        previous_response = state.to_response()
        current_step = action.text
        
        score = self.step_judge.judge(QUESTION=problem, PREVIOUS_STEPS=previous_response, CURRENT_STEP=current_step)
        if score is None: # parsing error
            score = 0
            
        return score
    
    def eval_solution(self, answer: str, solutions: list[Solution]):
        ground_truth = extract_boxed_answer(answer)
        
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