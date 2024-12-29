import os
import torch
from jsonargparse import CLI
from omegaconf import OmegaConf
from collections import Counter

from llm_reasoning.llm import get_model
from llm_reasoning.task import get_task
from llm_reasoning.llm.base import LLM, InferenceConfig
from llm_reasoning.task.base import Solution

def get_complete_path(task, root):
    terminals = []
        
    def dfs(node, total_reward):
        if node.state.is_terminal():
            solution = Solution(text=node.state.to_response())
            task_reward = task.eval_solution(node.state.answer, [solution])
            terminals.append({"node": node, "total_reward": total_reward, "task_reward": task_reward})
            return
        visited_childred = [child for child in node.children if child.visits > 0]
        for child in visited_childred:
            dfs(child, total_reward + child.reward)
            
    dfs(root, 0)
        
    return terminals

def generate_pretrain_data(task, root):
    terminals = get_complete_path(task, root)
    print(Counter([res["task_reward"] for res in terminals]))
    
    

def main(config_file: str):
    configs = OmegaConf.load(config_file)
    assert configs.policy.name == "mcts"
    
    model = LLM(model_name=configs.model.model_name)
    inference_config = InferenceConfig(**configs.model)
    task = get_task(model, inference_config, configs.task)
    
    auxiliary_file = os.path.join(configs.experiment.exp_dir, "auxiliary.pth")
    auxiliary = torch.load(auxiliary_file, map_location="cpu")
    
    save_dir = os.path.join(configs.experiment.exp_dir, "pretrain_data")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(100):
        root = auxiliary[i]["root"]
        generate_pretrain_data(task, root)
        
    

if __name__ == "__main__":
    CLI(main)