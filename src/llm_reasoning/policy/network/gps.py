import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.utils import to_dense_batch, scatter, to_dense_adj
from torch_geometric.data import Data, Batch
from torch_geometric.utils.num_nodes import maybe_num_nodes

def collate_fn(batch, num_rw_steps):
    ksteps = list(range(1, num_rw_steps+1))
    data_list = []
    offset = [0]
    for ex in batch:
        num_nodes = len(ex["node_features"])
        offset.append(num_nodes)
        node_embed = torch.stack([f["embedding"] for f in ex["node_features"]]) # num_node x feat_dim
        node_depth = torch.tensor([f["depth"] for f in ex["node_features"]], dtype=torch.long) # num_node
        
        edge_step = torch.tensor([f["step"] for f in ex["edge_features"]], dtype=torch.float) # num_edge
        edge_prob = torch.tensor([f["logprob"] for f in ex["edge_features"]], dtype=torch.float) # num_edge
        edge_attr = torch.stack([edge_step, edge_prob], dim=1)
        
        edge_index = ex["edge_index"].long().view(2,-1)
        
        RWSE = get_rw_landing_probs(ksteps, edge_index, num_nodes=num_nodes)
        
        data = Data(
            x=node_embed,
            node_depth=node_depth,
            edge_index=edge_index,
            edge_attr=edge_attr,
            RWSE=RWSE,
            current_node_idx=ex["current_node_idx"]
        )
        
        data_list.append(data)
        
    batch = Batch.from_data_list(data_list)
    
    # current_node_idx need to be manually offset
    offset = torch.tensor(offset[:-1], dtype=torch.long)
    batch.current_node_idx = batch.current_node_idx + offset
    
    return {
        "batch": batch
    }
    

class GPSPolicy(nn.Module):
    def __init__(self,
                 node_dim, 
                 max_depth,
                 edge_dim,
                 hidden_dim, 
                 pe_dim, 
                 num_rw_steps,
                 gps_layers,
                 num_heads, 
                 dropout,
                 attn_dropout, 
                 layer_norm,
                 batch_norm,
                 num_actions
    ):
        super().__init__()
        
        self.num_rw_steps = num_rw_steps
        
        # node feature
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.depth_embed = nn.Embedding(max_depth+1, hidden_dim)
        self.rwse = RWSENodeEncoder(pe_dim, num_rw_steps)
        self.node_feature = nn.Linear(hidden_dim * 2 + pe_dim, hidden_dim)
        
        # edge feature
        self.edge_feature = nn.Linear(edge_dim, hidden_dim)
        
        self.gps_layers = nn.Sequential(*[
            GPSLayer(
                dim_h=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm
            )
            for _ in range(gps_layers)
        ])
        
        self.constraints = nn.Linear(num_actions, hidden_dim)
        self.actor = nn.Linear(hidden_dim * 2, num_actions)
        self.critic = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, batch, action_masks):
        node_embed = self.node_proj(batch.x)
        depth_embed = self.depth_embed(batch.node_depth)
        batch.x = torch.cat([node_embed, depth_embed], dim=1)
        batch = self.rwse(batch)
        batch.x = self.node_feature(batch.x)
        batch.edge_attr = self.edge_feature(batch.edge_attr)
        batch = self.gps_layers(batch)
        
        # Extract the current node's embedding
        current_node_embedding = batch.x[batch.current_node_idx] # num_graph x feat_dim
        
        # Encode the action constraints
        c = self.constraints(action_masks.float())
        
        # policy inputs
        policy_inputs = torch.cat([current_node_embedding, c], dim=1)
        
        # Compute action logits and value prediction
        logits = self.actor(policy_inputs)
        inf_logits = torch.ones_like(logits) * -1e6
        logits = torch.where(action_masks, logits, inf_logits)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)
        vpred = self.critic(policy_inputs).squeeze(1)
        
        return {
            'dist': dist,
            'act': act,
            'logp': logp,
            'vpred': vpred,
        }
        

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None, num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


class RWSENodeEncoder(nn.Module):
    def __init__(self, dim_pe, num_rw_steps):
        super().__init__()
        
        self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        
    def forward(self, batch):
        pos_enc = batch.RWSE # num_nodes x num_rw_steps
        pos_enc = self.pe_encoder(pos_enc)
        
        batch.x = torch.cat([batch.x, pos_enc], dim=1)
        
        return batch
    
   
class GPSLayer(nn.Module):
    """
    GINE + Transformer
    """
    def __init__(self, dim_h, num_heads, dropout, attn_dropout, layer_norm, batch_norm):
        super().__init__()
        
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        
        # Local message-passing model.
        gin_nn = nn.Sequential(*[
            pygnn.Linear(dim_h, dim_h),
            nn.ReLU(),
            pygnn.Linear(dim_h, dim_h)
        ])
        self.local_model = pygnn.GINEConv(gin_nn)
        
        # Global attention transformer-style model.
        self.self_attn = nn.MultiheadAttention(dim_h, num_heads, dropout=attn_dropout, batch_first=True)
        
        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
            
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        
        # Feed Forward block.
        self.ff_block = nn.Sequential(*[
            nn.Linear(dim_h, dim_h * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout)
        ])
        
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)

        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        
    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection
        
        h_out_list = []
        
        # Local MPNN with edge attributes.
        h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection.
        
        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        
        h_out_list.append(h_local)
        
        # Multi-head attention.
        h_dense, mask = to_dense_batch(h, batch.batch)
        h_attn = self.self_attn(h_dense, h_dense, h_dense, attn_mask=None, key_padding_mask=~mask, need_weights=False)[0]
        h_attn = h_attn[mask]
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch.batch)
        
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        
        h_out_list.append(h_attn)
        
        # Combine local and global outputs.
        h = sum(h_out_list)
        
        # Feed Forward block.
        h = h + self.ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        
        return batch