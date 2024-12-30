from omegaconf import OmegaConf

from llm_reasoning.policy.cot import CoT
from llm_reasoning.policy.step_cot import StepCoT
from llm_reasoning.policy.mcts import MCTS
from llm_reasoning.policy.pgts import PGTS
from llm_reasoning.policy.ada_pgts import AdaPGTS
from llm_reasoning.policy.iter_pgts import IterPGTS

from llm_reasoning.task.base import Task

def get_policy(env: Task, policy_config: OmegaConf):
    if policy_config.name == "cot":
        return CoT.from_config(env, policy_config)
    elif policy_config.name == "step_cot":
        return StepCoT.from_config(env, policy_config)
    elif policy_config.name == "mcts":
        return MCTS.from_config(env, policy_config)
    elif policy_config.name == "pgts":
        return PGTS.from_config(env, policy_config)
    elif policy_config.name == "ada_pgts":
        return AdaPGTS.from_config(env, policy_config)
    elif policy_config.name == "iter_pgts":
        return IterPGTS.from_config(env, policy_config)
    else:
        raise NotImplementedError()