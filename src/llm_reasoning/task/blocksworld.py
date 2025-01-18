import os
import re
import json
import yaml
import random
import torch
import math
import numpy as np
import logging
import tempfile
from copy import deepcopy
from retry import retry
from tarski.io import PDDLReader
from typing import Optional
from pydantic import BaseModel, ConfigDict
from collections import defaultdict, Counter

from llm_reasoning.task.base import Task, Solution, Action, State
from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig
from llm_reasoning.judge.base import BaseJudge
from llm_reasoning.judge.llm_judge import LLMJudge
from llm_reasoning.judge.post_processor import LikertScaleProcessor
from llm_reasoning.judge.likelihood_judge import LikelihoodJudge

logger = logging.getLogger(__name__)

CHAT_TEMPLATE = """
{%- for message in messages %}
    {{- message.content}}
{%- endfor %}
""".strip()
    
ACTION_STEP_SEPARATOR = "\n"
    
class BWStateUpdateException(Exception):
    pass
    
class BWExample(BaseModel):
    instance_file: str
    plan_code: str
    init: str
    goal: str
    plan: str
    
    def evaluate(self, data_dir, config_file, domain_file, generated_plan) -> bool:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        problem = parse_instance(domain_file, self.instance_file)
        
        plan, readable_plan = format_blocksworld_plan(generated_plan, config, problem)
        logger.debug(f"Generated Plan:\n{generated_plan}\n\nFormatted Plan:\n{plan}\n\nReadable Plan:\n{readable_plan}\n\n")
        
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write(plan)
            plan_file = f.name 
        correctness = validate_plan(os.path.join(data_dir, "planner_tools/VAL"), domain_file, self.instance_file, plan_file)
        os.remove(plan_file)
        
        return correctness
        
    
class BWDataset(BaseModel):
    data_dir: str
    config_file: str
    domain_file: str
    examples: list[BWExample]
    
    @classmethod
    def load(cls, data_dir, config_file, domain_file, data_file):
        with open(data_file, "r") as f:
            data_list = json.load(f)
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        examples = []
        for ex in data_list:
            instance_file = os.path.join(data_dir, "instances", ex["instance_file"])
            problem = parse_instance(domain_file, instance_file)
            init, goal, plan = parse_blocksworld(problem, config, plan_code=ex["plan_code"], return_plan=True)
            
            examples.append(BWExample(
                instance_file=instance_file,
                plan_code=ex["plan_code"],
                init=init,
                goal=goal,
                plan=plan,
            ))
       
        return cls(
            data_dir=data_dir,
            config_file=config_file,
            domain_file=domain_file,
            examples=examples
        )

        
class BWAction(Action):
    text: str
    
    def __str__(self):
        return self.text
    

class BWState(State):
    problem: list[dict]
    question: str
    answer: BWExample
    blocks_state: list[str]
    trace: list[BWAction]
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return "".join([self.question, self.to_response()])
    
    def __hash__(self):
        return hash(str(self))
    
    def to_messages(self):
        messages = self.problem.copy()
        if self.trace:
            assistant_prefill = self.to_response() + ACTION_STEP_SEPARATOR
            messages.append({"role": "assistant", "content": assistant_prefill})
            
        return messages
    
    def __eq__(self, other):
        return isinstance(other, BWState) and str(self) == str(other)
    
    def to_response(self):
        return ACTION_STEP_SEPARATOR.join([action.text for action in self.trace]).lstrip()
    
    def is_terminal(self):
        if not self.trace:
            return False
        goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", self.answer.goal)
        
        return goal_check(goals, self.blocks_state[-1])[0]
    
        
class BlocksWorld(Task):
    data: BWDataset
    prompts: dict
    icl_templates: list[str]
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
        
        # construct icl templates
        def get_icl_demo(examples):
            icl = prompts["intro"] + "\n".join([
                "[STATEMENT]\nAs initial conditions I have that, " + example["init"] + \
                ".\nMy goal is to have that " + example["goal"] + \
                ".\n\nMy plan is as follows:\n\n[PLAN]" + example["plan"]
                for example in examples
            ])
            icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
            return icl
        
        ## choose icl examples
        rng = np.random.RandomState(0)
        examples = rng.choice(prompts["example_pool"], task_config.num_shot, replace=False).tolist()
        
        ## adaptive icl demonstration
        icl_templates = [get_icl_demo(examples)]
        for _ in range(5):
            new_examples = []
            for example in examples:
                if len(example["states"]) > 1:
                    new_examples.append({
                        "init": example["states"][0],
                        "goal": example["goal"],
                        "plan": "\n" + "\n".join(example["plan"].split("\n")[3:]),
                        "states": example["states"][1:]
                    })
                else:
                    new_examples.append(example)
            examples = deepcopy(new_examples)
            icl_templates.append(get_icl_demo(examples))
            
        ## change default chat_template
        inference_config = inference_config.model_copy(update={"model_config": ConfigDict(frozen=False)}, deep=True)
        inference_config.chat_template = CHAT_TEMPLATE
        
        return cls(
            data=data,
            prompts=prompts,
            icl_templates=icl_templates,
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
        
        # get question
        question = self.prompts["intro"]
        ## initial state
        question += "\n[STATEMENT]\n"
        question += f"As initial conditions I have that, {test_example.init}."
        ## goal condition
        question += f"\nMy goal is to have that {test_example.goal}."
        ## plan prefix
        question += f"\n\nMy plan is as follows:\n\n[PLAN]\n"
        
        # get question embedding
        embedding = self.model.get_prompt_embedding(
            messages=[{"role": "user", "content": question}],
            inference_config=InferenceConfig(chat_template=CHAT_TEMPLATE)
        )
        
        # get messages
        icl_template = self.icl_templates[0]
        problem = icl_template.replace(
            "<init_state>", test_example.init
        ).replace(
            "<goals>", test_example.goal
        ).replace(
            "<action>", ""
        )
        messages = [{"role": "user", "content": problem}]
        
        return BWState(
            problem=messages,
            question=question,
            answer=test_example,
            blocks_state=[test_example.init],
            trace=[],
            embedding=embedding,
        )
    
    @retry(BWStateUpdateException, tries=5)
    def _update_blocks_state(self, blocks_state: str, action: str):
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            raise ValueError("Invalid action")
        
        world_update_prompt = self.prompts[key].format(blocks_state, action.capitalize() + ".")
        world_output: list[LLMResponse] = self.model.batch_call(
            batch_messages=[[{"role": "user", "content": world_update_prompt}]],
            inference_config=InferenceConfig(
                temperature=self.inference_config.temperature,
                top_p=self.inference_config.top_p,
                max_tokens=128,
                stop_sequences=["\n"],
                chat_template=CHAT_TEMPLATE
            )
        )
        return apply_change(world_output[0].text, blocks_state)
    
    def transition(self, state: BWState, action: BWAction) -> BWState:
        embedding = self.model.get_prompt_embedding(
            messages=[{"role": "user", "content": state.question + state.to_response() + ACTION_STEP_SEPARATOR + action.text}],
            inference_config=InferenceConfig(chat_template=CHAT_TEMPLATE)
        )
        
        return BWState(
            problem=state.problem,
            question=state.question,
            answer=state.answer,
            blocks_state=state.blocks_state + [self._update_blocks_state(state.blocks_state[-1], action.text)],
            trace=state.trace + [action],
            embedding=embedding,
        )
        
    def _get_action_logprob(self, state: BWState, action: BWAction):
        num_steps = len(state.trace)
        icl_template = self.icl_templates[num_steps // 2]
        blocks_state = state.blocks_state[(num_steps // 2) * 2]
        if num_steps % 2 == 0:
            previous_action = ""
        else:
            previous_action = state.trace[-1].text + "\n"
            
        prefix = icl_template.replace(
            "<init_state>", blocks_state
        ).replace(
            "<goals>", state.answer.goal
        ).replace(
            "<action>", previous_action
        )
        action_logprob = self.model.get_answer_probs(
            messages=[{"role": "user", "content": prefix}],
            answer_candidates=[action.text],
            inference_config=InferenceConfig(chat_template=CHAT_TEMPLATE),
            normalize=False
        )
        
        return action_logprob[0]
        
    async def step(self, state: BWState, action: BWAction) -> tuple[BWState, float, bool, dict]:
        next_state = self.transition(state, action)
        
        done = next_state.is_terminal()
        
        info = {}
        
        reward = 0.0
        if 'action_logprob' in self.reward_coeff:
            action_logprob = self._get_action_logprob(state, action)
            reward += math.exp(action_logprob) * self.reward_coeff["action_logprob"] # exp to avoid negative reward
            info["action_logprob"] = math.exp(action_logprob)
        if 'action_quality' in self.reward_coeff:
            action_quality = self.eval_action(state, action)
            reward += action_quality * self.reward_coeff["action_quality"]
            info["action_quality"] = action_quality
        if 'goal_reward' in self.reward_coeff:
            goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", state.answer.goal)
            goal_reward = goal_check(goals, next_state.blocks_state[-1])[1]
            reward += goal_reward * self.reward_coeff["goal_reward"]
            info["goal_reward"] = goal_reward
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
        return [
            BWAction(text=action)
            for action in generate_all_actions(state.blocks_state[-1])
        ]
    
    def eval_state(self, state: BWState):
        raise NotImplementedError()
    
    def eval_action(self, state: BWState, action: BWAction):
        self_eval_prompt = self.prompts["self-eval"].replace(
            "<init_state>", state.blocks_state[-1]
        ).replace(
            "<goals>", state.answer.goal
        ).replace(
            "<action>", action.text
        )
        self_eval_logprob = self.model.get_answer_probs(
            messages=[{"role": "user", "content": self_eval_prompt}], 
            answer_candidates=["good"],
            inference_config=InferenceConfig(chat_template=CHAT_TEMPLATE),
            normalize=False
        )
        
        return math.exp(self_eval_logprob[0])
    
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
    
# utils
def parse_instance(domain_file, instance_file):
    pddl_reader = PDDLReader(raise_on_error=True)
    pddl_reader.parse_domain(domain_file)
    problem = pddl_reader.parse_instance(instance_file)
    
    return problem

def parse_problem(problem, config, shuffle):

    def get_sorted(init_atoms):
        return sorted(
            init_atoms,
            key=lambda x: x.symbol.name + " " + " ".join(
                [subterm.name for subterm in x.subterms]),
        )

    def parse(init_goal_preds, OBJS):
        TEXT = ""
        predicates = []

        init_goal_preds = list(init_goal_preds)
        for atom in init_goal_preds:
            objs = []
            for subterm in atom.subterms:
                objs.append(OBJS[subterm.name])
            predicates.append(config["predicates"][atom.symbol.name].format(*objs))
        if len(predicates) > 1:
            TEXT += ", ".join(predicates[:-1]) + f" and {predicates[-1]}"
        else:
            TEXT += predicates[0]
        return TEXT

    OBJS = config["encoded_objects"]

    init_atoms = get_sorted(problem.init.as_atoms())
    goal_preds = (get_sorted(problem.goal.subformulas) if hasattr(
        problem.goal, "subformulas") else [problem.goal])

    if shuffle:
        random.shuffle(init_atoms)
        random.shuffle(goal_preds)
    # ----------- INIT STATE TO TEXT ----------- #
    INIT = parse(init_atoms, OBJS)

    # ----------- GOAL TO TEXT ----------- #
    GOAL = parse(goal_preds, OBJS)

    return INIT, GOAL

def parse_blocksworld(problem, config, plan_code, return_plan, shuffle=False):
    """
    Function to make a blocksworld instance into human-readable format
    """
    OBJS = config["encoded_objects"]
    
    # ----------- PARSE THE PROBLEM ----------- #
    INIT, GOAL = parse_problem(problem, config, shuffle)
    
    # ----------- PLAN TO TEXT ----------- #
    PLAN = ""
    if return_plan:
        assert plan_code, "plan_code is not given, cannot parse plan"
        PLAN = "\n"
        plan = plan_code.split("\n")[:-1]
        for action in plan:
            action = action.strip("(").strip(")")
            act_name, objs = action.split(" ")[0], action.split(" ")[1:]
            objs = [OBJS[obj] for obj in objs]
            PLAN += config["actions"][act_name].format(*objs) + "\n"
        PLAN += "[PLAN END]\n"
        
    # keywords translation
    INIT = INIT.replace("-", " ").replace("ontable", "on the table")
    GOAL = GOAL.replace("-", " ").replace("ontable", "on the table")
    PLAN = PLAN.replace("-", " ").replace("ontable", "on the table")
    
    return INIT, GOAL, PLAN

def get_ordered_objects(object_names, line):
    objs = []
    pos = []
    for obj in object_names:
        if obj in line:
            objs.append(obj)
            pos.append(line.index(obj))
    sorted_zipped_lists = sorted(zip(pos, objs))
    return [el for _, el in sorted_zipped_lists]

def format_blocksworld_plan(generated_plan, config, problem):
    action_set = problem.actions
    # ----------- GET DICTIONARIES ----------- #
    LD = config["encoded_objects"]  # Letters Dictionary
    BD = {v: k for k, v in LD.items()}  # Blocks Dictionary

    # ----------- GET RAW AND TEXT-FORMATTED ACTIONS AND OBJECTS ----------- #
    actions_params_dict = dict(action_set.items())
    raw_actions = list(action_set.keys())
    text_actions = [x.replace("-", " ") for x in raw_actions]

    generated_plan = generated_plan.lower().strip()
    for raw_action, text_action in zip(raw_actions, text_actions):
        generated_plan = generated_plan.replace(text_action, raw_action)

    object_names = [x.lower() for x in LD.values()]

    # ----------- GET PLAN FROM TEXT ----------- #
    plan = ""
    readable_plan = ""
    lines = [line.strip() for line in generated_plan.split("\n")]
    for line in lines:
        if "[COST]" in line:
            break
        # Extracting actions
        action_list = [action in line.split() for action in raw_actions]
        if sum(action_list) == 0:
            continue

        action = raw_actions[np.where(action_list)[0][0]]
        # Extracting Objects
        n_objs = len(actions_params_dict[action].parameters.vars())
        objs = get_ordered_objects(object_names, line)
        if len(objs) != n_objs:
            continue
        readable_objs = [obj.replace(" block", "") for obj in objs]
        objs = [BD[x] for x in objs]
        readable_action = "({} {})".format(action, " ".join(readable_objs[:n_objs + 1]))
        action = "({} {})".format(action, " ".join(objs[:n_objs + 1]))

        plan += f"{action}\n"
        readable_plan += f"{readable_action}\n"

    return plan, readable_plan

def validate_plan(val_dir, domain_file, instance_file, plan_file):
    cmd = f"{val_dir}/validate {domain_file} {instance_file} {plan_file}"
    response = os.popen(cmd).read()
    logger.debug(f"VAL response: {response}")
    
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')

    if "Plan valid" in response:
        return True
    return False

def goal_check(goals, blocks_state):
    """Check if the goals are met and return the percentage of goals met

    :param goals: goals
    :param blocks_state: current blocks state
    """
    meetings = [g in blocks_state for g in goals]
    if sum(meetings) == len(meetings):
        return True, 1.0
    return False, sum(meetings) / len(meetings)

def apply_change(change, state):
    """Apply the predicted change to the state
    
    :param change: predicted change
    :param state: current state
    """
    if "and the " in state and ", and the" not in state:
        state = state.replace("and the ", ", and the ")
    states = state.split(", ")
    states = [s.strip()[4:].strip(".") if s.strip().startswith("and ") else s.strip().strip(".") for s in states]
    changes = change.lower().strip().strip(".").split(", ")
    for c in changes:
        if c.startswith("and "):
            c = c[4:]
        success = 0
        if c.startswith("the hand"):
            match = re.search(r"was (.*?) (?:and is now|and)", c)
            old = match.group(1).strip() if match else ""
            if "and is now" in c:
                new = c.split("now")[1].strip()
            elif "and" in c:
                new = c.split("and")[1].strip()
            else:
                new = ""  
            for idx in range(len(states)):
                if ("hand is " + old) in states[idx]:
                    states[idx] = states[idx].replace(old, new)
                    success += 1
        else:
            colors = re.findall(r"the (\w+) block", c)
            if len(colors) == 0:
                logger.warning("Error: zero-colors")
                logger.warning(c)                
                raise BWStateUpdateException()
            color = colors[0]
            if c.startswith(f"the {color} block"):
                subj = f"{color} block"
                if "no longer" in c:
                    old = c.split("no longer")[1].strip()
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = ""
                            success += 1
                elif "was" in c and "now" in c:
                    old = c.split("was")[1].split(" and")[0].strip()
                    new = c.split("now")[1].strip()
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = states[idx].replace(old, new)
                            success += 1
                elif "now" in c:
                    new = c.split("now")[1].strip()
                    states.append("the " + color + " block is " + new)
                    success += 1
            else:
                logger.warning("Error: not recognized")
                logger.warning(c)
                raise BWStateUpdateException()
        
        if success == 0:
            logger.warning("Error: no successful change")
            logger.warning(c)
            logger.warning(states)
            raise BWStateUpdateException()
    states = [s for s in states if s != ""]
    priority_states = []
    for s in states:
        if "have that" in s:
            priority_states.append(0)
        elif "clear" in s:
            priority_states.append(1)
        elif "in the hand" in s:
            priority_states.append(1)
        elif "the hand is" in s:
            priority_states.append(2)
        elif "on top of" in s:
            priority_states.append(3)
        elif "on the table" in s:
            priority_states.append(4)
        else:
            logger.warning("Error: unknown state")
            logger.warning(s)
            raise BWStateUpdateException()
    sorted_states = [x.strip() for _, x in sorted(zip(priority_states, states))]
    sorted_states[-1] = "and " + sorted_states[-1]
    return ", ".join(sorted_states) + "."

def generate_all_actions(state):
    """Generate all possible actions from the current state

    :param state: current state
    """
    return_list = []
    if "hand is empty" in state:
        block = re.findall("the [a-z]{0,10} block is clear", state)
        block_color = [
            re.search("the ([a-z]{0,10}) block is clear", b).group(1)
            for b in block
        ]
        for c in block_color:
            if f"the {c} block is on the table" in state:
                return_list.append(f"pick up the {c} block")
            else:
                c_ = re.search(
                    f"the {c} block" + " is on top of the ([a-z]{0,10}) block",
                    state).group(1)
                return_list.append(
                    f"unstack the {c} block from on top of the {c_} block")
    else:
        c = re.search("is holding the ([a-z]{0,10}) block", state).group(1)
        block = re.findall("the [a-z]{0,10} block is clear", state)
        clear_color = [
            re.search("the ([a-z]{0,10}) block is clear", b).group(1)
            for b in block
        ]
        for c_ in clear_color:
            return_list.append(f"stack the {c} block on top of the {c_} block")
        return_list.append(f"put down the {c} block")
    return return_list