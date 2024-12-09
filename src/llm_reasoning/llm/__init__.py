from omegaconf import OmegaConf

from llm_reasoning.llm.api import APIModel
from llm_reasoning.llm.vllm import vLLMModel
from llm_reasoning.llm.huggingface import HuggingFaceModel

def get_model(model_config: OmegaConf):
    if model_config.model_source == "api":
        return APIModel(model_name=model_config.model_name)
    elif model_config.model_source == "vllm":
        return vLLMModel(model_name=model_config.model_name, port=model_config.port)
    elif model_config.model_source == "huggingface":
        return HuggingFaceModel(model_name=model_config.model_name, batch_size=model_config.batch_size, device=model_config.device)
    else:
        raise NotImplementedError()