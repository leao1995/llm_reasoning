import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

def collate_fn(batch, tokenizer):
    trajectories = []
    for ex in batch:
        trajectory = f"Question: {ex['question']}\n\n"
        for i, node in enumerate(ex["reasoning_path"]):
            trajectory += f"Step {i}: {node['text']}\n"
        trajectories.append(trajectory.strip())
        
    return tokenizer(trajectories, padding=True, truncation=True, return_tensors="pt")

class SLMPolicy(nn.Module):
    
    SUPPORTED_MODELS = [
        "distilbert/distilbert-base-uncased"
    ]
    
    def __init__(self, base_model, freeze_backbone, hidden_dim, dropout, num_actions):
        super().__init__()
        
        assert base_model in self.SUPPORTED_MODELS
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.base_model = AutoModel.from_pretrained(base_model)
        self.projection = nn.Sequential(
            nn.Linear(self.base_model.config.dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.constraints = nn.Linear(num_actions, hidden_dim)
        self.actor = nn.Linear(hidden_dim * 2, num_actions)
        self.critic = nn.Linear(hidden_dim * 2, 1)
        
        if freeze_backbone:
            self.freeze_backbone()
        
    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, action_masks):
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_state = base_output.last_hidden_state[:, 0] #[CLS] token
        hidden_state = self.projection(hidden_state)
        c = self.constraints(action_masks.float())
        policy_inputs = torch.cat([hidden_state, c], dim=1)
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
