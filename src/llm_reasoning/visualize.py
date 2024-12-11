import torch
from pyvis.network import Network

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
    
info_file = "exp/qwen2.5_7b/gsm8k/mcts_debug/auxiliary.pth"
info = torch.load(info_file)
TreeVisualizer(["state", "reward", "Q", "visits"]).visualize(info[0]["root"], "debug.html")