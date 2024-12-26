from omegaconf import OmegaConf

from llm_reasoning.llm.api import APIModel
from llm_reasoning.llm.vllm_chat import vLLMChatModel
from llm_reasoning.llm.vllm_inference import vLLMInferenceModel
from llm_reasoning.llm.huggingface import HuggingFaceModel
from llm_reasoning.llm.vllm import vLLMModel

def get_model(model_config: OmegaConf):
    if model_config.model_source == "api":
        return APIModel(model_name=model_config.model_name)
    elif model_config.model_source == "vllm_chat":
        return vLLMChatModel(model_name=model_config.model_name, port=model_config.port)
    elif model_config.model_source == "vllm_inference":
        return vLLMInferenceModel(model_name=model_config.model_name, port=model_config.port, device=model_config.device)
    elif model_config.model_source == "huggingface":
        return HuggingFaceModel(model_name=model_config.model_name, batch_size=model_config.batch_size, device=model_config.device)
    elif model_config.model_source == "vllm":
        return vLLMModel(draft_model_name=model_config.draft_model_name, model_name=model_config.model_name, max_seq_len=model_config.max_seq_len, devices=model_config.devices)
    else:
        raise NotImplementedError()