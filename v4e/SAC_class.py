import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

import random
from collections import deque

class _SACActor(nn.Module):
    def __init__(self):
        super(_SACActor, self).__init__()
        
        # 影像特徵提取
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
        
        # 儀表板特徵提取
        self.dashboard_embed = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 獨立的輸出頭 (不可共用 nn.Linear)
        self.mu_head = nn.Linear(128, 3)
        self.log_std_head = nn.Linear(128, 3)

    def forward(self, map_img, dashboard_vec):
        # 影像分支
        map_feat = self.map_embed(map_img)
        # 數值分支
        dash_feat = self.dashboard_embed(dashboard_vec)
        
        # 特徵對齊拼接 (維度 1)
        embedding = torch.cat([map_feat, dash_feat], dim=1) # (Batch, 128)
        
        mu = self.mu_head(embedding)
        log_std = self.log_std_head(embedding)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, map_img, dashboard_vec):
        # 呼叫修正後的 forward
        mu, log_std = self.forward(map_img, dashboard_vec)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample() 
        
        action_tanh = torch.tanh(x_t)
        
        # 計算 Log Prob 並進行 tanh 修正
        log_prob = dist.log_prob(x_t) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        
        # 分離與縮放動作 [Steer, Gas, Brake]
        steer = action_tanh[:, 0:1]
        gas_brake = (action_tanh[:, 1:] + 1.0) / 2.0
        action = torch.cat([steer, gas_brake], dim=1)
        
        # 針對 Gas/Brake 維度進行線性縮放修正 (log 2)
        # 這裡用一個簡單的加法處理
        correction = torch.zeros_like(log_prob)
        correction[:, 1:] = torch.log(torch.tensor(2.0))
        log_prob += correction

        return action, log_prob.sum(1, keepdim=True)

class _QNetwork(nn.Module):
    def __init__(self):
        super(_QNetwork, self).__init__()
        
        # 影像特徵提取
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
        
        # 儀表板特徵提取
        self.dashboard_embed = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 動作特徵提取
        self.action_embed = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Q 值融合頭 (輸入：影像 64 + 數值 64 + 動作 64 = 131)
        self.q_head = nn.Sequential(
            nn.Linear(64 + 64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 輸出單一標量 Q 值
        )

    def forward(self, map_img, dash_vec, action):
        m_feat = self.map_embed(map_img)
        d_feat = self.dashboard_embed(dash_vec)
        a_feat = self.action_embed(action)
        
        return self.q_head(torch.cat([m_feat, d_feat, a_feat], dim=1))

class _SACDoubleCritic(nn.Module):
    def __init__(self):
        super(_SACDoubleCritic, self).__init__()
        # SAC 核心：建立兩個完全獨立的 Q 網路
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
        # 樹的節點總數 = 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # 實際儲存經驗資料的陣列
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """向上更新父節點的值"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """根據隨機值 s 尋找對應的葉子節點"""
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
        """加入新經驗與初始優先權"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """更新葉子節點的優先權，並連帶更新整棵樹"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """根據抽樣權重尋找經驗"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]  

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.25, beta=0.4, beta_increment=0.00001):
        self.tree = _SumTree(capacity)
        self.alpha = alpha  # 決定優先權的主導程度 (0: 完全隨機, 1: 完全靠 TD Error)
        self.beta = beta    # 修正重要性採樣偏誤的參數 (隨著訓練逐漸調整到 1.0)
        self.beta_increment = beta_increment
        self.epsilon = 0.01 # 防止優先權變成 0 的極小值

    def push(self, state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done):
        # 新加入的經驗一律給予目前樹中「最高的優先權」，確保它至少會被抽到一次
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        
        data = (state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done)
        self.tree.add(max_p, data)

    def sample(self, batch_size):
        b_idx, b_memory, IS_weights = [], [], []
        
        # 將總優先權區間等分成 batch_size 個區段，在每個區段內均勻抽樣
        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 計算用來歸一化 IS Weights 的最大可能權重 (防止權重爆炸)
        leaf_start = self.tree.capacity - 1
        leaf_end = leaf_start + self.tree.n_entries
        valid_priorities = self.tree.tree[leaf_start:leaf_end]

        p_min = np.min(valid_priorities) / self.tree.total_priority
        p_min = max(p_min, 1e-5)

        max_weight = (p_min * self.tree.n_entries) ** (-self.beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, p, data = self.tree.get(s)
            
            # 計算重要性採樣權重
            sampling_probabilities = p / self.tree.total_priority
            weight = (sampling_probabilities * self.tree.n_entries) ** (-self.beta)
            IS_weights.append(weight / max_weight)
            
            b_idx.append(idx)
            b_memory.append(data)

        # 解包資料
        state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done = zip(*b_memory)
        
        return (np.array(state_img), np.array(dash_vec), np.array(action, dtype=np.float32), 
                np.array(reward, dtype=np.float32), np.array(next_state_img), 
                np.array(next_dash_vec), np.array(done, dtype=np.float32),
                np.array(b_idx), np.array(IS_weights, dtype=np.float32))

    def batch_update(self, tree_idx, abs_errors):
        """根據新算出來的 TD Error 更新樹中的優先權"""
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, 10.0) # 防止梯度過大
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.n_entries

class SACAgent:
    def __init__(self, device):
        self.device = device
        self.actor = _SACActor().to(device)
        self.critic = _SACDoubleCritic().to(device)
        self.target_critic = _SACDoubleCritic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 優化器
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 自動溫度調整 (Entropy Alpha)
        self.target_entropy = -3.0 # 目錄維度 -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        self.gamma = 0.99
        self.tau = 0.005

    def update_parameters(self, buffer, batch_size):
        # 1. 採樣時多拿 tree_idx 和 IS_weights
        s_img, s_dash, a, r, ns_img, ns_dash, d, tree_idx, is_weights = buffer.sample(batch_size)        
        
        # 轉換為 Tensor 的邏輯保持原樣...
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

        # 2. 更新 Critic (計算 MSE Loss 時乘上重要性採樣權重)
        curr_q1, curr_q2 = self.critic(s_img, s_dash, a)
        
        # 乘上權重修正偏誤
        critic_loss1 = (weights * F.mse_loss(curr_q1, target_q, reduction='none')).mean()
        critic_loss2 = (weights * F.mse_loss(curr_q2, target_q, reduction='none')).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        # 3. 計算用於更新 SumTree 優先權的 TD Error (通常取兩個 Critic 的平均或最小值之絕對值)
        with torch.no_grad():
            td_error = (torch.abs(curr_q1 - target_q) + torch.abs(curr_q2 - target_q)) / 2.0
            td_error = td_error.cpu().numpy().flatten()
        
        # 將新的 TD Error 餵回 Buffer 更新樹節點
        buffer.batch_update(tree_idx, td_error)

        # 4. 更新 Actor 與 Alpha 的邏輯保持原樣...
        new_a, log_prob = self.actor.sample(s_img, s_dash)
        q1_new, q2_new = self.critic(s_img, s_dash, new_a)
        actor_loss = (self.log_alpha.exp() * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # Soft Update
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
