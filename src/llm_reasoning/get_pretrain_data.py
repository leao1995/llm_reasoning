import os
import torch
from jsonargparse import CLI
from omegaconf import OmegaConf

from llm_reasoning.llm import get_model
from llm_reasoning.task import get_task
from llm_reasoning.llm.base import InferenceConfig

def generate_bc_data(root):
    pass

def main(config_file: str):
    configs = OmegaConf.load(config_file)
    assert configs.policy.name == "mcts"
    
    model = get_model(configs.model)
    inference_config = InferenceConfig(**configs.model)
    task = get_task(model, inference_config, configs.task)
    
    auxiliary_file = os.path.join(configs.experiment.exp_dir, "auxiliary.pth")
    auxiliary = torch.load(auxiliary_file)
    
    save_dir = os.path.join(configs.experiment.exp_dir, "pretrain_data")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(task.size):
        pass
    

if __name__ == "__main__":
    CLI(main)