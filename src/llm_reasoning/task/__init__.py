from omegaconf import OmegaConf

from llm_reasoning.task.gsm8k import GSM8K
from llm_reasoning.task.math import Math
from llm_reasoning.task.aqua import Aqua
from llm_reasoning.task.strategyqa import StrategyQA
from llm_reasoning.task.gpqa import GPQA
from llm_reasoning.task.prontoqa import ProntoQA
from llm_reasoning.task.blocksworld import BlocksWorld

from llm_reasoning.llm.base import LLM, InferenceConfig

def get_task(model: LLM, inference_config: InferenceConfig, task_config: OmegaConf):
    if task_config.name == "gsm8k":
        return GSM8K.from_config(model=model, inference_config=inference_config, task_config=task_config)
    elif task_config.name == "math":
        return Math.from_config(model=model, inference_config=inference_config, task_config=task_config)
    elif task_config.name == "aqua":
        return Aqua.from_config(model=model, inference_config=inference_config, task_config=task_config)
    elif task_config.name == "strategyqa":
        return StrategyQA.from_config(model=model, inference_config=inference_config, task_config=task_config)
    elif task_config.name == "gpqa":
        return GPQA.from_config(model=model, inference_config=inference_config, task_config=task_config)
    elif task_config.name == "prontoqa":
        return ProntoQA.from_config(model=model, inference_config=inference_config, task_config=task_config)
    elif task_config.name == "blocksworld":
        return BlocksWorld.from_config(model=model, inference_config=inference_config, task_config=task_config)
    else:
        raise NotImplementedError()