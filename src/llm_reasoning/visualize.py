import os
import torch
from pyvis.network import Network
from jsonargparse import CLI
from omegaconf import OmegaConf

class TreeVisualizer:
    def __init__(self, properties: list[str]):
        self.properties = properties
    
    def get_properties(self, node, verbose=False):
        prop = {}
        for k in self.properties:
            if k == "state" and not verbose:
                prop[k] = getattr(node, k).to_response()
            else:
                prop[k] = getattr(node, k)
        
        return "\n\n".join(f"{k}: {v}" for k, v in prop.items())

    def visualize(self, root, output_file):
        net = Network(directed=True)
        node_id_map = {}
        
        # dfs to add nodes
        def dfs(node):
            for child in node.children:
                node_id_map[child] = len(node_id_map)
                net.add_node(node_id_map[child], title=self.get_properties(child))
                net.add_edge(node_id_map[node], node_id_map[child])
                dfs(child)
        
        node_id_map[root] = 0
        net.add_node(0, title=self.get_properties(root, verbose=True))
        dfs(root)
        
        net.save_graph(output_file)
        
def main(config_file: str, num_plots=10):
    configs = OmegaConf.load(config_file)
    if configs.policy.name == "mcts":
        properties = ["state", "reward", "Q", "visits"]
    elif configs.policy.name == "pgts":
        properties = ["state", "reward", "node_id"]
    else:
        raise NotImplementedError()
    
    visalizer = TreeVisualizer(properties)
    
    info_file = os.path.join(configs.experiment.exp_dir, "auxiliary.pth")
    info = torch.load(info_file)
    
    save_dir = os.path.join(configs.experiment.exp_dir, "visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_plots):
        save_path = os.path.join(save_dir, f"ex_{i}.html")
        visalizer.visualize(info[i]["root"], save_path)

if __name__ == "__main__":
    CLI(main)