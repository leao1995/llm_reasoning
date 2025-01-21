import os
import asyncio
import logging
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
import random
import torch
from statistics import mean
from collections import namedtuple

from llm_reasoning.task.base import State, Action, Task, Solution
from llm_reasoning.policy.base import Policy
from llm_reasoning.policy.network import gnn, xformer, san, gps, prompting, slm
from llm_reasoning.policy.utils import plot_reward, ppo_advantage, ReplayBuffer

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state: State, reward: float, info: dict, parent=None, depth=0):
        self.state = state
        self.reward = reward # reward from the action that leads to this state
        self.info = info
        self.parent = parent
        self.children = []
        self.visits = 0
        self.depth = depth
        self.child_idx = 0
        # additional properties for policy
        self.node_id: int = None
        self.trace: list[tuple[int, State]] = []
        self.node_features: list[dict] = []
        self.edge_features: list[dict] = []
        self.edge_index: list[list[int]] = []
    
    def add_child(self, child: 'Node'):
        self.children.append(child)
        
    def select_child(self):
        return self.children[self.child_idx]
    
    def has_unvisited_child(self):
        return len(self.children) == 0 or self.child_idx < len(self.children) - 1
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def __hash__(self):
        return hash(self.state)
    

class TreeSearchEnv:
    def __init__(self, env: Task, max_steps: int, max_depth: int, max_breadth: int, edge_type: str, action_costs: dict, extra_bonus: dict):
        self.env = env
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.edge_type = edge_type
        self.action_costs = action_costs
        self.extra_bonus = extra_bonus
        
    def init(self, root: Node):
        root.node_id = 0
        root.trace.append((-1, root.state))
        root.node_features.append({"embedding": root.state.embedding, "depth": root.depth})
        root.edge_features = []
        root.edge_index = []
        
        # reset current node
        self.root = root
        self.node = root
        # reset num_steps
        self.num_steps = 0
        self.terminals = []
        # reset global graph representations
        self.node_id = 0
        self.trace = [(-1, root.state)]
        self.node_features = [{"embedding": root.state.embedding, "depth": root.depth}]
        self.edge_features = []
        self.edge_index = []
        
    async def _expand(self):
        actions: list[Action] = self.env.propose_actions(self.node.state, self.max_breadth)
        # sort based on action logprob
        actions = sorted(actions, key=lambda x: x.log_prob or 0, reverse=True)
        steps = [asyncio.create_task(self.env.step(self.node.state, action)) for action in actions]
        outputs = await asyncio.gather(*steps)
            
        for next_state, reward, _, info in outputs:
            if next_state.is_terminal():
                assert reward-info["task_reward"] * self.env.reward_coeff["task_reward"] >= 0
                child_node = Node(next_state, reward-info["task_reward"] * self.env.reward_coeff["task_reward"], info, self.node, self.node.depth+1)
            else:
                child_node = Node(next_state, reward, info, self.node, self.node.depth+1)
            self.node.add_child(child_node)
            
    async def _continue(self):
        '''
        continue from the current node to its child node
        if the current node is not expanded, we need to first expand it
        '''
        if not self.node.children:
            await self._expand()
        
        next_node = self.node.select_child()
        interm_reward = next_node.reward
        reward = interm_reward - self.action_costs["continue"]
        
        return next_node, reward
    
    def _branch(self):
        '''
        go the the next sibling node
        if no more siblings to explore, keep it the same
        '''
        parent = self.node.parent
        if parent.child_idx < len(parent.children) - 1:
            parent.child_idx += 1
        next_node = parent.select_child()
        interm_reward = next_node.reward - self.node.reward
        
        reward = interm_reward - self.action_costs["branch"]
        
        return next_node, reward
    
    def _backtrack(self, steps):
        '''
        backtrack to previous nodes
        policy will guarantee the number of steps is valid
        '''
        cur_node = self.node
        interm_reward = 0
        while steps > 0:
            interm_reward -= cur_node.reward
            cur_node = cur_node.parent
            steps -= 1
        parent = cur_node.parent
        if parent.child_idx < len(parent.children) - 1:
            parent.child_idx += 1
        next_node = parent.select_child()
        interm_reward += next_node.reward - cur_node.reward
        
        reward = interm_reward - self.action_costs["backtrack"]
        
        return next_node, reward
    
    def _terminate(self):
        '''
        terminate the searching process
        '''
        next_node = self.node
        
        if "task_reward" in next_node.info: # terminate at leaf node
            reward = next_node.info["task_reward"] * self.env.reward_coeff["task_reward"] - self.action_costs["terminate"]
        else: # terminate when reach depth limit
            reward = - self.action_costs["terminate"]
        
        return next_node, reward
    
    def _get_edges(self, node: Node):
        if self.edge_type == "directed":
            edge_idx = [(node.parent.node_id, node.node_id)]
            edge_feat = [{"step": node.depth - node.parent.depth, "logprob": node.info["action_logprob"]}]
        elif self.edge_type == "undirected":
            edge_idx = [(node.parent.node_id, node.node_id), (node.node_id, node.parent.node_id)]
            edge_feat = [{"step": node.depth - node.parent.depth, "logprob": node.info["action_logprob"]}, 
                         {"step": node.parent.depth - node.depth, "logprob": node.info["action_logprob"]}]
        elif self.edge_type == "path_directed":
            edge_idx = [(node.parent.node_id, node.node_id)]
            edge_feat = [{"step": node.depth - node.parent.depth, "logprob": node.info["action_logprob"]}]
            parent = node.parent.parent
            while parent is not None:
                edge_idx.append((parent.node_id, node.node_id))
                edge_feat.append({"step": node.depth - parent.depth, "logprob": None})
                parent = parent.parent
        elif self.edge_type == "path_undirected":
            edge_idx = [(node.parent.node_id, node.node_id), (node.node_id, node.parent.node_id)]
            edge_feat = [{"step": node.depth - node.parent.depth, "logprob": node.info["action_logprob"]}, 
                         {"step": node.parent.depth - node.depth, "logprob": node.info["action_logprob"]}]
            parent = node.parent.parent
            while parent is not None:
                edge_idx.extend([(parent.node_id, node.node_id), (node.node_id, parent.node_id)])
                edge_feat.extend([{"step": node.depth - parent.depth, "logprob": None},
                                  {"step": parent.depth - node.depth, "logprob": None}])
                parent = parent.parent
        elif self.edge_type == "full":
            edge_idx = [(node.parent.node_id, node.node_id), (node.node_id, node.parent.node_id)]
            edge_feat = [{"step": node.depth - node.parent.depth, "logprob": node.info["action_logprob"]}, 
                         {"step": node.parent.depth - node.depth, "logprob": node.info["action_logprob"]}]
            for i in range(node.node_id):
                if i != node.parent.node_id:
                    edge_idx.extend([(i, node.node_id), (node.node_id, i)])
                    edge_feat.extend([{"step": None, "logprob": None},
                                      {"step": None, "logprob": None}])
        else:
            raise NotImplementedError()
        
        return edge_idx, edge_feat
    
    def _path_diversity(self, node: Node):
        if not self.terminals:
            return 0.0
        # find lowest common ancestor
        heights = []
        for term_node, path in self.terminals:
            cur_node = node
            while cur_node:
                if cur_node in path:
                    heights.append(node.depth - cur_node.depth)
                    break
                cur_node = cur_node.parent
        min_height = min(heights)
        
        return (min_height / self.max_depth) * self.extra_bonus["path_diversity"]
    
    async def step(self, action: int):
        if action == 0: # continue
            next_node, reward = await self._continue()
        elif action == 1:
            next_node, reward = self._branch()
        elif action == self.max_depth + 1: # in total depth+2 actions
            next_node, reward = self._terminate()
        else:
            next_node, reward = self._backtrack(action-1) # the number of backtract steps equals to action-1
                    
        self.num_steps += 1
        
        next_node.visits += 1
        
        # update node properties. branch and backtrack may hit the breadth limit, therefore next node might have been visited
        if next_node.node_id is None:
            # update graph representation
            self.node_id += 1
            next_node.node_id = self.node_id # do not change node id if it's visited before
            self.trace.append((action, next_node.state))
            self.node_features.append({"embedding": next_node.state.embedding, "depth": next_node.depth})
            edge_index, edge_features = self._get_edges(next_node)
            self.edge_features.extend(edge_features)
            self.edge_index.extend(edge_index)
        
        # record graph representation as node properties, these properties will be updated even for seen nodes
        next_node.trace = self.trace.copy()
        next_node.node_features = self.node_features.copy()
        next_node.edge_features = self.edge_features.copy()
        next_node.edge_index = self.edge_index.copy()
        
        # reward for path diversity
        if next_node.state.is_terminal() and "path_diversity" in self.extra_bonus:
            diversity_bonus = self._path_diversity(next_node)
            reward += diversity_bonus
            # save the path
            path = set()
            cur_node = next_node
            while cur_node:
                path.add(cur_node)
                cur_node = cur_node.parent
            self.terminals.append((next_node, path))
        
        self.node = next_node
        
        done = action == self.max_depth + 1 or self.num_steps >= self.max_steps
        
        return next_node, reward, done, {}
    
    
class TreeSearchPolicy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    policy_dir: str
    policy_name: str
    policy_type: str
    policy_network: torch.nn.Module
    policy_device: str
    
    max_depth: int
    num_actions: int
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        # 0: continue
        # 1: branch
        # 2 - max_depth: bracktrack 1 - max_depth-1 steps
        # max_depth+1: terminate
        num_actions = policy_config.depth_limit + 2
        
        if policy_config.policy_type == "gnn":
            policy_network = gnn.GNNPolicy(
                node_dim=policy_config.node_dim, 
                hidden_dim=policy_config.hidden_dim, 
                num_actions=num_actions
            ).to(policy_config.device)
        elif policy_config.policy_type == "xformer":
            policy_network = xformer.XformerPolicy(
                node_dim=policy_config.node_dim,
                hidden_dim=policy_config.hidden_dim,
                num_actions=num_actions,
                num_layers=policy_config.num_layers,
                num_heads=policy_config.num_heads,
                dropout=policy_config.dropout
            ).to(policy_config.device)
        elif policy_config.policy_type == "san":
            policy_network = san.SANPolicy(
                gamma=policy_config.gamma,
                node_dim=policy_config.node_dim,
                max_depth=policy_config.depth_limit,
                edge_dim=policy_config.edge_dim,
                hidden_dim=policy_config.hidden_dim,
                pe_dim=policy_config.pe_dim,
                pe_layers=policy_config.pe_layers,
                san_layers=policy_config.num_layers,
                num_heads=policy_config.num_heads,
                dropout=policy_config.dropout,
                layer_norm=policy_config.layer_norm,
                batch_norm=policy_config.batch_norm,
                residual=policy_config.residual,
                full_graph=policy_config.full_graph,
                num_actions=num_actions,
            ).to(policy_config.device)
        elif policy_config.policy_type == "gps":
            policy_network = gps.GPSPolicy(
                node_dim=policy_config.node_dim,
                max_depth=policy_config.depth_limit,
                edge_dim=policy_config.edge_dim,
                hidden_dim=policy_config.hidden_dim,
                pe_dim=policy_config.pe_dim,
                num_rw_steps=policy_config.num_rw_steps,
                gps_layers=policy_config.num_layers,
                num_heads=policy_config.num_heads,
                dropout=policy_config.dropout,
                attn_dropout=policy_config.attn_dropout,
                layer_norm=policy_config.layer_norm,
                batch_norm=policy_config.batch_norm,
                num_actions=num_actions,
            ).to(policy_config.device)
        elif policy_config.policy_type == "prompting":
            policy_network = prompting.LLMPolicy(
                model=env.model,
                inference_config=env.inference_config,
                num_actions=num_actions,
            )
        elif policy_config.policy_type == "slm":
            policy_network = slm.SLMPolicy(
                base_model=policy_config.base_model,
                freeze_backbone=policy_config.freeze_backbone,
                hidden_dim=policy_config.hidden_dim,
                dropout=policy_config.dropout,
                num_actions=num_actions
            ).to(policy_config.device)
        else:
            raise NotImplementedError
        
        logger.info(f"Policy {policy_config.policy_type} has {sum(p.numel() for p in policy_network.parameters() if p.requires_grad)} parameters")

        return cls(
            policy_dir=policy_config.policy_dir,
            policy_name=policy_config.policy_name,
            policy_type=policy_config.policy_type,
            policy_network=policy_network,
            policy_device=policy_config.device,
            max_depth=policy_config.depth_limit,
            num_actions=num_actions,
        )
        
    def save(self, step=None):
        if step is not None:
            os.makedirs(os.path.join(self.policy_dir, f"ckpt_{step}"), exist_ok=True)
            torch.save(self.policy_network.state_dict(), os.path.join(self.policy_dir, f"ckpt_{step}/{self.policy_name}.pth"))
            logger.info(f"saving checkpoint at step {step}")
        else:
            os.makedirs(self.policy_dir, exist_ok=True)
            torch.save(self.policy_network.state_dict(), os.path.join(self.policy_dir, f"{self.policy_name}.pth"))
            logger.info(f"saving to {self.policy_dir}")
    
    def load(self):
        if os.path.exists(os.path.join(self.policy_dir, f"{self.policy_name}.pth")):
            self.policy_network.load_state_dict(torch.load(os.path.join(self.policy_dir, f"{self.policy_name}.pth"), map_location="cpu"))
            self.policy_network.to(self.policy_device)
            logger.info(f"loading from {self.policy_dir}")
        else:
            logger.warning(f"model weights does not exist, using random weights")
    
    def get_trainable_parameters(self):
        return self.policy_network.parameters()
    
    def set_status(self, status):
        if status == "train":
            self.policy_network.train()
        else:
            self.policy_network.eval()
            
    def _prepare_inputs(self, nodes: list[Node]):
        def _get_path(node):
                path = []
                current = node
                while current.parent:
                    path.append({"node_id": current.node_id, "depth": current.depth, "text": current.state.trace[-1].text})
                    current = current.parent
                return path[::-1]
        
        batch = [
            {
                "node_features": node.node_features,
                "edge_features": node.edge_features,
                "edge_index": torch.tensor(node.edge_index).t(),
                "current_node_idx": node.node_id,
            }
            for node in nodes
        ]
        if self.policy_type == "gnn":
            batch_inputs = gnn.collate_fn(batch)
            batch_inputs = {k: v.to(self.policy_device) for k, v in batch_inputs.items()}
        elif self.policy_type == "xformer":
            batch_inputs = xformer.collate_fn(batch)
            batch_inputs = {k: v.to(self.policy_device) for k, v in batch_inputs.items()}
        elif self.policy_type == "san":
            batch_inputs = san.collate_fn(batch, max_freqs=10)
            batch_inputs = {k: v.to(self.policy_device) for k, v in batch_inputs.items()}
        elif self.policy_type == "gps":
            batch_inputs = gps.collate_fn(batch, self.policy_network.num_rw_steps)
            batch_inputs = {k: v.to(self.policy_device) for k, v in batch_inputs.items()}
        elif self.policy_type == "prompting":
            batch = [
                {
                    "question": next((msg["content"] for msg in reversed(node.state.to_messages()) if msg["role"] == "user")),
                    "reasoning_path": _get_path(node)
                }
                for node in nodes
            ]
            batch_inputs = prompting.collate_fn(batch)
        elif self.policy_type == "slm":
            batch = [
                {
                    "question": next((msg["content"] for msg in reversed(node.state.to_messages()) if msg["role"] == "user")),
                    "reasoning_path": _get_path(node)
                }
                for node in nodes
            ]
            batch_inputs = slm.collate_fn(batch, self.policy_network.tokenizer)
            batch_inputs = {k: v.to(self.policy_device) for k, v in batch_inputs.items()}
        else:
            raise NotImplementedError()
        
        return batch_inputs
    
    def _prepare_constraints(self, nodes: list[Node]):
        action_masks = []
        for node in nodes:
            mask = torch.zeros(self.num_actions).bool()
            # depth specific constraints
            if node.depth == 0: # cannot branch the root node, only valid action is continue
                mask[:1] = 1
            elif node.depth == 1: # no previous steps to backtrack, can continue or branch
                mask[:2] = 1
            elif node.depth == self.max_depth: # leaf node cannot expand further, can branch or backtrack
                mask[1:] = 1
            else: # continue and branch are valid, and can backtrack depth-1 steps
                mask[:node.depth+1] = 1
            # breadth specific constraints
            if node.parent is not None and not node.parent.has_unvisited_child():
                mask[1] = 0 # node does not have sibling nodes, cannot branch
            cur_node = node
            for backtrack_step in range(1, node.depth):
                cur_node = cur_node.parent
                if not cur_node.parent.has_unvisited_child():
                    mask[backtrack_step+1] = 0 # cannot backtrack
            # terminal node cannot expand further
            if node.state.is_terminal():
                mask[0] = 0
            # terminate is valid only when the node is a terminal node
            mask[-1] = 1 if node.state.is_terminal() else 0
            # special case: reached depth limit and no available nodes to explore
            if not torch.any(mask):
                mask[-1] = 1
            action_masks.append(mask)
        action_masks = torch.stack(action_masks).to(self.policy_device)
        
        return {
            "action_masks": action_masks
        }
        
    def __call__(self, nodes: list[Node]):
        batch_inputs = self._prepare_inputs(nodes)
        batch_constraints = self._prepare_constraints(nodes)
        
        return self.policy_network(**batch_inputs, **batch_constraints)
    

class AdaPGTS(Policy):
    env: Task
    policy: TreeSearchPolicy
    search_action_costs: dict[str, float]
    extra_bonus: dict[str, float]
    breadth_limit: int
    depth_limit: int
    max_search_steps: int
    edge_type: str
    num_chains: int
    
    @classmethod
    def from_config(cls, env: Task, policy_config: OmegaConf):
        assert env.inference_config.temperature > 0, "greedy decoding cannot produce multiple reasoning chains"
        
        policy = TreeSearchPolicy.from_config(env, policy_config)
        policy.load()
        policy.set_status("eval")
        
        return cls(
            env=env,
            policy=policy,
            search_action_costs=dict(policy_config.search_action_costs),
            extra_bonus=dict(policy_config.extra_bonus),
            breadth_limit=policy_config.breadth_limit,
            depth_limit=policy_config.depth_limit,
            edge_type=policy_config.edge_type,
            max_search_steps=policy_config.max_search_steps,
            num_chains=policy_config.num_chains,
        )
        
    async def _rollout(self, state: State):
        env = TreeSearchEnv(
            env=self.env,
            max_steps=self.max_search_steps,
            max_depth=self.depth_limit,
            max_breadth=self.breadth_limit,
            edge_type=self.edge_type,
            action_costs=self.search_action_costs,
            extra_bonus=self.extra_bonus,
        )
        root = Node(state, 0, {})
        env.init(root)
        node, done = env.node, False
        trajectory = []
        total_reward = 0.0
        while not done:
            res = self.policy([node])
            next_node, reward, done, info = await env.step(res["act"][0].item())
            trajectory.append({
                "node": [node],
                "act": res["act"].data,
                "logp": res['logp'].data,
                "vpred": res['vpred'].data,
                "reward": torch.tensor([reward]).to(res['vpred']),
            })
            total_reward += reward
            node = next_node
            
        return trajectory, total_reward, node
    
    async def _run(self, state: State) -> tuple[list[Solution], dict]:
        rollouts = [asyncio.create_task(self._rollout(state)) for _ in range(self.num_chains)]
        outputs = await asyncio.gather(*rollouts)
        
        solutions = []
        trajectory = []
        total_reward = []
        dummy_node = Node(None, None, {}, None)
        for traj, traj_reward, final_node in outputs:
            solutions.append(Solution(text=final_node.state.to_response(), weight=traj_reward))
            root = traj[0]["node"][0]
            root.parent = dummy_node
            dummy_node.children.append(root)
            trajectory.append(traj)
            total_reward.append(traj_reward)
        
        info = {
            "root": dummy_node,
            "trajectory": trajectory,
            "total_reward": total_reward,
        }
        
        return solutions, info
    
    def run(self, state: State) -> tuple[list[Solution], dict]:
        return asyncio.run(self._run(state))
    
    def _learn(self, buffer, optimizer, num_iters, batch_size, vf_weight, ent_weight, ratio_clip, grad_norm):
        for _ in range(num_iters):
            actor_loss = []
            critic_loss = []
            entropy = []
            for minibatch in buffer.loop(batch_size):
                forward = self.policy(minibatch["node"])
                # actor loss
                ratio = (forward['dist'].log_prob(minibatch["act"]) - minibatch["logp"]).exp().float()
                surr1 = ratio * minibatch["adv"]
                surr2 = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip) * minibatch["adv"]
                clip_loss = -torch.min(surr1, surr2).mean()
                # critic loss
                value = forward['vpred']
                vf_loss = (minibatch["v_target"] - value).pow(2).mean()
                # regularization
                ent_loss = forward['dist'].entropy().mean()
                # optimize
                loss = clip_loss + vf_weight * vf_loss - ent_weight * ent_loss
                optimizer.zero_grad()
                loss.backward()
                if grad_norm:  # clip large gradient
                    torch.nn.utils.clip_grad_norm_(self.policy.get_trainable_parameters(), max_norm=grad_norm)
                optimizer.step()
                # log
                actor_loss.append(clip_loss.data.cpu().item())
                critic_loss.append(vf_loss.data.cpu().item())
                entropy.append(ent_loss.data.cpu().item())
            
            logger.info(f"actor_loss: {mean(actor_loss)} critic_loss: {mean(critic_loss)} entropy: {mean(entropy)}")
            
    async def _collect(self, buffer, training_config):
        rollouts = []
        for i in range(training_config.num_rollout_per_step):
            idx = random.choice(range(self.env.size))
            init_state = self.env.init(idx)
            rollouts.append(asyncio.create_task(self._rollout(init_state)))
        outputs = await asyncio.gather(*rollouts)
        
        total_reward = 0
        for i, (traj, traj_reward, _) in enumerate(outputs):
            traj = ppo_advantage(traj, training_config.ppo_gamma, training_config.gae_lambda)
            buffer.add(traj, batch_size=1)
            total_reward += traj_reward
            logger.info(f"rollout {i} - traj_reward {traj_reward}")
        
        return total_reward / training_config.num_rollout_per_step
    
    def train(self, training_config: OmegaConf):
        self.policy.set_status("train")
        optimizer = torch.optim.Adam(self.policy.get_trainable_parameters(), lr=training_config.lr)

        # replay buffer
        BufferRecord = namedtuple("BufferRecord", ["node", "act", "logp", "vpred", "reward", "adv", "v_target"])
        buffer = ReplayBuffer(capacity=training_config.buffer_size, tuple_class=BufferRecord)
        
        # training
        best_reward = -1e6
        reward_hist = []
        for step in range(training_config.training_steps):
            total_reward = asyncio.run(self._collect(buffer, training_config))
            reward_hist.append(total_reward)
            self._learn(
                buffer, optimizer,
                training_config.training_iters_per_step, 
                training_config.training_minibatch_size,
                training_config.vf_weight,
                training_config.ent_weight,
                training_config.ratio_clip,
                training_config.grad_norm
            )
            if total_reward >= best_reward:
                best_reward = total_reward
                self.policy.save()
                logger.info(f"step {step} best_reward: {best_reward}")
            if step % 50 == 0:
                plot_reward(reward_hist, 10, os.path.join(self.policy.policy_dir, f"{self.policy.policy_name}.png"))
            if hasattr(training_config, "save_freq") and (step+1) % training_config.save_freq == 0:
                self.policy.save(step=step)
                
        if training_config.save_last:
            self.policy.save()
            
        plot_reward(reward_hist, 10, os.path.join(self.policy.policy_dir, f"{self.policy.policy_name}.png"))
        
        torch.save(reward_hist, os.path.join(self.policy.policy_dir, "reward_history.pth"))