import torch
import random
from statistics import mean
from collections import namedtuple, deque
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt


def ppo_advantage(traj, gamma, gae_lambda):
    assert isinstance(traj, list)
    assert all(t['vpred'] is not None and t['reward'] is not None for t in traj)
    
    vpreds = torch.stack([t['vpred'] for t in traj]) # [H, B]
    reward = torch.stack([t['reward'] for t in traj]) # [H, B]
    td_errors = [reward[h] + gamma * vpreds[h+1] - vpreds[h] for h in range(len(traj)-1)]
    td_errors += [reward[-1] - vpreds[-1]]
    advs = []
    adv_so_far = 0.0
    for delta in td_errors[::-1]:
        adv_so_far = delta + gamma * gae_lambda * adv_so_far
        advs.append(adv_so_far)
    advs = torch.stack(advs[::-1]) # [H, B]
    returns = advs + vpreds # [H, B]
    
    for h in range(len(traj)):
        traj[h]["adv"] = advs[h]
        traj[h]["v_target"] = returns[h]
        
    return traj

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])

def _stack(items: list):
    if all(isinstance(item, torch.Tensor) for item in items):
        return torch.stack(items)
    else:
        return items

class ReplayBuffer:
    def __init__(self, capacity=None, tuple_class=Transition):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.tuple_class = tuple_class
        self.fields = tuple_class._fields
        
    def add(self, trajectory: list[dict], batch_size):
        if self.capacity is None:
            self.buffer.clear()
            
        for b in range(batch_size):
            for traj in trajectory:
                record = {
                    k: v[b]
                    for k, v in traj.items()
                }
                self.buffer.append(self.tuple_class(**record))
                
    def _reformat(self, indices):
        return {
            field_name: _stack([getattr(self.buffer[i], field_name) for i in indices])
            for field_name in self.fields
        }
                
    def sample(self, batch_size):
        if len(self.buffer) >= batch_size:
            idxs = random.sample(range(len(self.buffer)), k=batch_size)
        else:
            idxs = random.choices(range(len(self.buffer)), k=batch_size)
            
        return self._reformat(idxs)
    
    def loop(self, batch_size):
        indices = []
        for i in random.sample(range(len(self.buffer)), k=len(self.buffer)):
            indices.append(i)
            if len(indices) == batch_size:
                yield self._reformat(indices)
                indices = []
            
        if len(indices) > 0:
            yield self._reformat(indices)
        
    @property
    def size(self):
        return len(self.buffer)

def plot_reward(reward_hist, window_size, save_path):
    reward_averaged = [mean(reward_hist[:i+1][-window_size:]) for i in range(len(reward_hist))]
    
    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot(range(len(reward_hist)), reward_hist)
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Reward")
    
    axs[1].plot(range(len(reward_averaged)), reward_averaged)
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Averaged Reward")
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)