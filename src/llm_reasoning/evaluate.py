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
        filename=os.path.join(configs.experiment.exp_dir, 'evaluate.log'),
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
    
def evaluate(configs: OmegaConf, task: Task, policy: Policy):
    outputs = []
    auxiliary = []
    metrics = []
    configs.experiment.max_task_size = configs.experiment.max_task_size if hasattr(configs.experiment, "max_task_size") else np.inf
    for index in range(min(task.size, configs.experiment.max_task_size)):
        init_state = task.init(index)
        solutions, info = policy.run(init_state)
        metric = task.eval_solution(init_state.answer, solutions)
        outputs.append(solutions)
        auxiliary.append(info)
        metrics.append(metric)
    torch.save(outputs, os.path.join(configs.experiment.exp_dir, "outputs.pth"))
    torch.save(auxiliary, os.path.join(configs.experiment.exp_dir, "auxiliary.pth"))
    torch.save(metrics, os.path.join(configs.experiment.exp_dir, "metrics.pth"))
    
    logging.info(f"metrics: {mean(metrics)}")
    
def main(config_file: str):
    configs = OmegaConf.load(config_file)
    setup_logging(configs)
    seed_everything(configs)
    
    model = get_model(configs.model)
    inference_config = InferenceConfig(**configs.model)
    task = get_task(model, inference_config, configs.task)
    policy = get_policy(task, configs.policy)
    
    evaluate(configs, task, policy)

if __name__ == "__main__":
    CLI(main)