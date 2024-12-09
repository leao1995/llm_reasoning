import os
import logging
import random
import torch
import numpy as np
from statistics import mean
from jsonargparse import CLI
from omegaconf import OmegaConf

from llm_reasoning.llm import get_model
from llm_reasoning.task import get_task
from llm_reasoning.policy import get_policy
from llm_reasoning.llm.base import InferenceConfig
from llm_reasoning.task.base import Task
from llm_reasoning.policy.base import Policy

logger = logging.getLogger(__name__)

def setup_logging(configs):
    logging.basicConfig(
        filename=os.path.join(configs.experiment.exp_dir, 'train.log'),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding='utf-8',
        level=logging.INFO
    )
    
def seed_everything(configs):
    random.seed(configs.experiment.seed)
    np.random.seed(configs.experiment.seed)
    torch.manual_seed(configs.experiment.seed)
    torch.cuda.manual_seed_all(configs.experiment.seed)
    os.environ["GLOBAL_RANDOM_SEED"] = str(configs.experiment.seed)
    
def main(config_file: str):
    configs = OmegaConf.load(config_file)
    setup_logging(configs)
    seed_everything(configs)
    
    model = get_model(configs.model)
    inference_config = InferenceConfig(**configs.model)
    task = get_task(model, inference_config, configs.task)
    policy = get_policy(task, configs.policy)
    policy.train(configs.training)

if __name__ == "__main__":
    CLI(main)