import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops, get_laplacian, to_scipy_sparse_matrix, to_undirected, add_self_loops
from torch_geometric.data import Data, Batch

def collate_fn(batch, max_freqs):
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
        
        # Laplacian
        L = to_scipy_sparse_matrix(
            *get_laplacian(to_undirected(
            edge_index=add_self_loops(edge_index, num_nodes=num_nodes)[0], 
            num_nodes=num_nodes
        ), normalization="sym", num_nodes=num_nodes))
        evals, evects = np.linalg.eigh(L.toarray())
        
        # Keep up to the maximum desired number of frequencies.
        idx = evals.argsort()[:max_freqs]
        evals, evects = evals[idx], np.real(evects[:, idx])
        evals = torch.from_numpy(np.real(evals)).clamp_min(0)
        # Normalize and pad eigen vectors.
        evects = torch.from_numpy(evects).float()
        evects = F.normalize(evects, p=2, dim=1, eps=1e-12)
        if num_nodes < max_freqs:
            EigVecs = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))
        else:
            EigVecs = evects
        # Pad and save eigenvalues.
        if num_nodes < max_freqs:
            EigVals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan')).unsqueeze(0)
        else:
            EigVals = evals.unsqueeze(0)
        EigVals = EigVals.repeat(num_nodes, 1).unsqueeze(2)
        
        data = Data(
            x=node_embed, 
            node_depth=node_depth, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            EigVals=EigVals, 
            EigVecs=EigVecs, 
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


class SANPolicy(nn.Module):
    def __init__(self, 
                 gamma, 
                 node_dim, 
                 max_depth,
                 edge_dim,
                 hidden_dim, 
                 pe_dim, 
                 pe_layers, 
                 san_layers, 
                 num_heads, 
                 dropout, 
                 layer_norm,
                 batch_norm,
                 residual,
                 full_graph, 
                 num_actions
    ):
        super().__init__()
        
        # node feature
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.depth_embed = nn.Embedding(max_depth+1, hidden_dim)
        self.lap_pe = LapPENodeEncoder(pe_dim, num_heads, pe_layers)
        self.node_feature = nn.Linear(hidden_dim * 2 + pe_dim, hidden_dim)
        
        # edge feature
        self.edge_feature = nn.Linear(edge_dim, hidden_dim)
        
        self.san_layers = nn.Sequential(*[
            SANLayer(
                gamma=gamma,
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                full_graph=full_graph,
                dropout=dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                residual=residual,
            )
            for _ in range(san_layers)
        ])
        
        self.constraints = nn.Linear(num_actions, hidden_dim)
        self.actor = nn.Linear(hidden_dim * 2, num_actions)
        self.critic = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, batch, action_masks):
        node_embed = self.node_proj(batch.x)
        depth_embed = self.depth_embed(batch.node_depth)
        batch.x = torch.cat([node_embed, depth_embed], dim=1)
        batch = self.lap_pe(batch)
        batch.x = self.node_feature(batch.x)
        batch.edge_attr = self.edge_feature(batch.edge_attr)
        batch = self.san_layers(batch)
        
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


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    
    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short, device=edge_index.device)
        adj = scatter(zero, idx, dim=0, dim_size=flattened_size, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    
    return edge_index_negative


class LapPENodeEncoder(nn.Module):
    def __init__(self, dim_pe, n_heads, n_layers):
        super().__init__()
        
        self.linear_A = nn.Linear(2, dim_pe)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe, nhead=n_heads, batch_first=True)
        self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError(f"Precomputed eigen values and vectors are required for {self.__class__.__name__}")
        
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs
        
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)
        
        # Permutation Equivariant Embedding    
        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0])
        
        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2), 0.)
        
        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe
        
        # Concatenate final PEs to input embedding
        batch.x = torch.cat([batch.x, pos_enc], dim=1)
        
        return batch
        

class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """
    
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias=False):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        
        if self.full_graph:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.fake_edge_emb = torch.nn.Embedding(1, in_dim)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        
    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[fake_edge_index[0]]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[fake_edge_index[1]]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1
            score_2 = torch.exp(score_2.sum(-1, keepdim=True).clamp(-5, 5))  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce='add')
        if self.full_graph:
            scatter(score_2, fake_edge_index[1], dim=0, out=batch.Z, reduce='add')
        
    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)
        
        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            E_2 = self.E_2(dummy_edge)
            
        V_h = self.V(batch.x)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out

    
class SANLayer(nn.Module):
    """GraphTransformerLayer from SAN.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0, layer_norm=False, batch_norm=True, residual=True):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(gamma=gamma, in_dim=in_dim, out_dim=out_dim // num_heads, num_heads=num_heads, full_graph=full_graph)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
        
    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)
        
        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        
        return batch
    