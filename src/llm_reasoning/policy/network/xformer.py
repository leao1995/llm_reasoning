import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

def collate_fn(batch):
    """
    Collate function for transformer policy with reasoning trees.

    Args:
        trees (list[dict]): A list of reasoning trees where each tree contains:
            - "node_features": Tensor of shape [num_nodes, feature_dim].
            - "edge_index": Tensor of shape [2, num_edges] (connectivity).
            - "current_node_idx": int, index of the current node for policy prediction.
    
    Returns:
        node_features (Tensor): Padded node features of shape [batch_size, max_nodes, feature_dim].
        attention_mask (Tensor): Attention mask of shape [batch_size, max_nodes, max_nodes].
        current_node_indices (Tensor): Indices of current nodes in the batch.
    """
    node_features_list = [tree["node_features"] for tree in batch]
    edge_index_list = [tree["edge_index"] for tree in batch]
    current_node_indices = torch.tensor([tree["current_node_idx"] for tree in batch], dtype=torch.long)
    
    # Compute max nodes in the batch
    max_nodes = max(features.size(0) for features in node_features_list)
    batch_size = len(batch)
    
    # Padded node features
    batch_node_features, _ = to_dense_batch(
        torch.cat(node_features_list),
        batch=torch.repeat_interleave(
            torch.arange(batch_size), 
            torch.tensor([f.size(0) for f in node_features_list])
        )
    )
    
    # Create attention mask for policy
    attention_mask = torch.ones((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    for i, (edge_index, node_features) in enumerate(zip(edge_index_list, node_features_list)):
        num_nodes = node_features.size(0)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj_matrix[edge_index[0], edge_index[1]] = True
        attention_mask[i, :num_nodes, :num_nodes] = ~adj_matrix
        
    return {
        "node_features": batch_node_features,
        "attention_mask": attention_mask,
        "current_node_idx": current_node_indices
    }

class XformerPolicy(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_actions, num_layers, num_heads, dropout=0.5):
        super().__init__()
        
        self.num_heads = num_heads
        
        self.projection = nn.Linear(node_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=4 * hidden_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features, attention_mask, current_node_idx, action_masks):
        """
        Args:
            node_features (Tensor): Node features of shape [batch_size, seq_len, feature_dim].
            attention_mask (Tensor): Binary mask for attention, shape [batch_size, seq_len, seq_len]. True indicates attention is not allowed.
            current_node_indices (Tensor): Indices of the current nodes in each batch, shape [batch_size].
            action_masks: Tensor of shape[batch_size, num_actions] 1 indicates valid actions

        Returns:
            Action and value prediction for the current node.
        """
        x = self.projection(node_features)
        
        # expand attention mask
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(-1, attention_mask.shape[1], attention_mask.shape[2])
        
        # transformer
        x = self.encoder(x, mask=attention_mask)
        
        # Select features corresponding to current nodes
        batch_indices = torch.arange(current_node_idx.size(0), device=current_node_idx.device)
        current_node_embedding = x[batch_indices, current_node_idx] # Shape: [batch_size, hidden_dim]
        
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