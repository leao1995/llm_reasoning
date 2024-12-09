import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

def collate_fn(batch):
    """
    Collate function for batching nodes, edges, and current node indices.

    Args:
        batch: A list of dictionaries, where each dictionary contains:
            - "node_features": Tensor of shape [num_nodes, feature_dim].
            - "edge_index": Tensor of shape [2, num_edges].
            - "current_node_idx": Index of the current reasoning node.

    Returns:
        collated_data: A dictionary containing:
            - "node_features": Batched tensor of node features.
            - "edge_index": Tensor of shape [2, total_num_edges] with adjusted indices.
            - "current_node_idx": Tensor of indices for the current reasoning nodes.
            - "batch_indices": Batch indices for nodes, useful for GNNs.
    """
    batched_node_features = []
    batched_edge_index = []
    batched_current_node_idx = []
    batch_indices = []

    node_offset = 0

    for graph_idx, graph_data in enumerate(batch):
        node_features = graph_data["node_features"]
        edge_index = graph_data["edge_index"]
        current_node_idx = graph_data["current_node_idx"]

        # Adjust edge indices for batching
        adjusted_edge_index = edge_index + node_offset
        batched_edge_index.append(adjusted_edge_index)

        # Accumulate node features
        batched_node_features.append(node_features)
        
        # Accumulate current node index with offset
        batched_current_node_idx.append(current_node_idx + node_offset)
        
        # Track batch assignments for each node
        num_nodes = node_features.shape[0]
        batch_indices.extend([graph_idx] * num_nodes)

        # Update node offset
        node_offset += num_nodes

    return {
        "node_features": torch.cat(batched_node_features, dim=0),  # [total_num_nodes, feature_dim]
        "edge_index": torch.cat(batched_edge_index, dim=1).long().view(2,-1),  # [2, total_num_edges]
        "current_node_idx": torch.tensor(batched_current_node_idx),  # [batch_size]
        "batch_indices": torch.tensor(batch_indices),  # [total_num_nodes]
    }

class GNNPolicy(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_actions):
        super().__init__()
        self.gcn1 = GCNConv(node_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features, edge_index, current_node_idx, batch_indices, action_masks):
        """
        Args:
            node_features: Tensor of shape [num_nodes, node_dim], embedding of each node.
            edge_index: Tensor of shape [2, num_edges], adjacency list of the tree.
            current_node_idx: Tensor of shape [batch_size] Index of the current reasoning node.
            action_masks: Tensor of shape[batch_size, num_actions] 1 indicates valid actions

        Returns:
            Action and value prediction for the current node.
        """
        # GCN layers
        x = self.gcn1(node_features, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)
        
        # Extract the current node's embedding
        current_node_embedding = x[current_node_idx]

        # Compute action logits and value prediction
        logits = self.actor(current_node_embedding)
        inf_logits = torch.ones_like(logits) * -1e6
        logits = torch.where(action_masks, logits, inf_logits)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)
        vpred = self.critic(current_node_embedding).squeeze(1)
        
        return {
            'dist': dist,
            'act': act,
            'logp': logp,
            'vpred': vpred,
        }