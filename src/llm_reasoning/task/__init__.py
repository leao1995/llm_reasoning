from omegaconf import OmegaConf

from llm_reasoning.task.gsm8k import GSM8K

from llm_reasoning.llm.base import LLM, InferenceConfig

def get_task(model: LLM, inference_config: InferenceConfig, task_config: OmegaConf):
    if task_config.name == "gsm8k":
        return GSM8K.from_config(model=model, inference_config=inference_config, task_config=task_config)
    else:
        raise NotImplementedError()