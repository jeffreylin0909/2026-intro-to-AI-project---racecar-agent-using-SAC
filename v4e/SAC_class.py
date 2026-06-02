import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.distributions import Normal

import numpy as np

import random
from collections import deque

class _SACActor(nn.Module):
    def __init__(self):
        super(_SACActor, self).__init__()
        
        # 影像特徵提取 (Img feature extractor)
        self.map_embed = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), # 40x48
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 20x24
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 10x12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 12, 512), 
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.ReLU()
        )
        
        # 儀表板特徵提取 (dashboard feature extractor)
        self.dash_embed = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # output heads
        self.mu_head = nn.Linear(128, 3)
        self.log_std_head = nn.Linear(128, 3)

    def forward(self, map_img, dash_vec):
        # extract features
        map_feat = self.map_embed(map_img)
        dash_feat = self.dash_embed(dash_vec)
        embed = torch.cat([map_feat, dash_feat], dim=1)
        
        mu = self.mu_head(embed)
        log_std = self.log_std_head(embed)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, map_img, dashboard_vec):
        mu, log_std = self.forward(map_img, dashboard_vec)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample() 
        
        action_tanh = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        
        # actions
        steer = action_tanh[:, 0:1]
        gas_brake = (action_tanh[:, 1:] + 1.0) / 2.0
        action = torch.cat([steer, gas_brake], dim=1)
        
        # scale correction
        correction = torch.zeros_like(log_prob)
        correction[:, 1:] = torch.log(torch.tensor(2.0))
        log_prob += correction

        return action, log_prob.sum(1, keepdim=True)

class _QNetwork(nn.Module):
    def __init__(self):
        super(_QNetwork, self).__init__()
        
        # 影像特徵提取 (Img feature extractor)
        self.map_embed = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), # 40x48
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 20x24
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 10x12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 12, 512), 
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.ReLU()
        )
        
        # 儀表板特徵提取 (dashboard feature extractor)
        self.dash_embed = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 動作特徵提取 (action feature extractor)
        self.action_embed = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(64 + 64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, map_img, dash_vec, action):
        map_feat = self.map_embed(map_img)
        dash_feat = self.dash_embed(dash_vec)
        act_feat = self.action_embed(action)

        embed = torch.cat([map_feat, dash_feat, act_feat], dim=1)
        q_val = self.q_head(embed)

        return q_val

class _SACDoubleCritic(nn.Module):
    def __init__(self):
        super(_SACDoubleCritic, self).__init__()
        # 建立兩個完全獨立的QNetwork (2 independent QNet)
        self.Q1 = _QNetwork()
        self.Q2 = _QNetwork()

    def forward(self, map_img, dash_vec, action):
        q1 = self.Q1(map_img, dash_vec, action)
        q2 = self.Q2(map_img, dash_vec, action)
        return q1, q2
    
class _ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done):
        self.buffer.append((state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done = zip(*batch)
        return (np.array(state_img), np.array(dash_vec), np.array(action, dtype=np.float32), 
                np.array(reward, dtype=np.float32), np.array(next_state_img), 
                np.array(next_dash_vec), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
    
class _SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # 樹的節點總數 (sumtree nodes)
        self.data = np.zeros(capacity, dtype=object) # actual data 
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        # 更新父節點的值 (update parent's value)
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        # 更新整棵樹 (update entire tree)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]  

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.25, beta=0.4, beta_increment=0.00001):
        self.tree = _SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01

    def push(self, state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done):
        # 確保新加入的至少會被抽到一次 (new item must be selected at least once)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0

        data = (state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done)
        self.tree.add(max_p, data)

    def sample(self, batch_size):
        batch_id = []
        batch_data = []
        weights = []

        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 防止權重爆炸 (prevent gradient explosion)
        leaf_start = self.tree.capacity - 1
        leaf_end = leaf_start + self.tree.n_entries
        valid_priorities = self.tree.tree[leaf_start:leaf_end]

        p_min = np.min(valid_priorities) / self.tree.total_priority
        p_min = max(p_min, 1e-5)

        max_weight = (p_min * self.tree.n_entries) ** (-self.beta)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            picked = random.uniform(low, high)

            idx, p, data = self.tree.get(picked)

            sampling_probabilities = p / self.tree.total_priority
            weight = (sampling_probabilities * self.tree.n_entries) ** (-self.beta)
            weights.append(weight / max_weight)

            batch_id.append(idx)
            batch_data.append(data)

        state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done = zip(*batch_data)
        
        return (np.array(state_img), np.array(dash_vec), np.array(action, dtype=np.float32), 
                np.array(reward, dtype=np.float32), np.array(next_state_img), 
                np.array(next_dash_vec), np.array(done, dtype=np.float32),
                np.array(batch_id), np.array(weights, dtype=np.float32))

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, 10.0) # 防止梯度過大 (cut excess gradient)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.n_entries

class SACAgent:
    def __init__(self, device, lr_opt=3e-4):
        self.device = device
        self.actor = _SACActor().to(device)
        self.critic = _SACDoubleCritic().to(device)
        self.target_critic = _SACDoubleCritic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr_opt)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr_opt)
        
        # Adaptive entropy
        self.target_entropy = -3.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr_opt)
        
        self.gamma = 0.99
        self.tau = 0.005

    def update_parameters(self, buffer, batch_size):
        # Sample Batch
        s_img, s_dash, a, r, ns_img, ns_dash, d, tree_idx, is_weights = buffer.sample(batch_size)        
        
        # Tensors setup
        s_img = torch.FloatTensor(s_img).permute(0, 3, 1, 2).to(self.device) / 255.0
        ns_img = torch.FloatTensor(ns_img).permute(0, 3, 1, 2).to(self.device) / 255.0

        scales = torch.FloatTensor([100.0, 230.0, 230.0, 230.0, 230.0, 1.0, 1.0, 1.0]).to(self.device)
        s_dash, ns_dash = torch.FloatTensor(s_dash).to(self.device)/scales, torch.FloatTensor(ns_dash).to(self.device)/scales
        a, r, d = torch.FloatTensor(a).to(self.device), torch.FloatTensor(r).unsqueeze(1).to(self.device), torch.FloatTensor(d).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device) # 新增權重 Tensor

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(ns_img, ns_dash)
            q1_t, q2_t = self.target_critic(ns_img, ns_dash, next_action)
            target_v = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_log_prob
            target_q = r + (1 - d) * self.gamma * target_v

        # update using PER weights
        curr_q1, curr_q2 = self.critic(s_img, s_dash, a)
        critic_loss1 = (weights * Func.mse_loss(curr_q1, target_q, reduction='none')).mean()
        critic_loss2 = (weights * Func.mse_loss(curr_q2, target_q, reduction='none')).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update PER priority
        with torch.no_grad():
            td_error = (torch.abs(curr_q1 - target_q) + torch.abs(curr_q2 - target_q)) / 2.0
            td_error = td_error.cpu().numpy().flatten()
        buffer.batch_update(tree_idx, td_error)

        # Update actor
        new_a, log_prob = self.actor.sample(s_img, s_dash)
        q1_new, q2_new = self.critic(s_img, s_dash, new_a)
        actor_loss = (self.log_alpha.exp() * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

    def save_checkpoint(self, checkpoint_path, episode, now_score, best_score):
        checkpoint = {
            'episode': episode,
            'now_score': now_score,
            'best_score': best_score,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_opt_state_dict': self.alpha_opt.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'])
        self.alpha_opt.load_state_dict(checkpoint['alpha_opt_state_dict'])
        return checkpoint['episode'], checkpoint['now_score'], checkpoint['best_score']
