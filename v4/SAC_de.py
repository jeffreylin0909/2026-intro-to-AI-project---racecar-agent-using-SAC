import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import os
import pickle
import shutil
import pandas as pd

import random
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    print(f"已設定亂數種子:{seed}")

SEED=7
set_seed(SEED)

class SACActor(nn.Module):
    def __init__(self):
        super(SACActor, self).__init__()
        #影像特徵提取
        self.map_embed=nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 12, 512), 
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.ReLU()
        )
        #儀表板特徵提取
        self.dashboard_embed=nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.mu_head=nn.Linear(128, 3)
        self.log_std_head=nn.Linear(128, 3)

    def forward(self, map_img, dashboard):
        x=self.map_embed(map_img) #map's feature
        y=self.dashboard_embed(dashboard) #dashboard's feature
        
        embedding=torch.cat([x, y], dim=1) 
        mu=self.mu_head(embedding)
        log_std=self.log_std_head(embedding)
        log_std=torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, map_img, dashboard):
        mu, log_std=self.forward(map_img, dashboard)
        std=log_std.exp()
        dist=Normal(mu, std)
        x_t=dist.rsample() 
        action_tanh=torch.tanh(x_t)
        
        log_prob=dist.log_prob(x_t) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        
        
        steer=action_tanh[:, 0:1]
        gas_brake=(action_tanh[:, 1:] + 1.0)/2.0
        action=torch.cat([steer, gas_brake], dim=1)
        
        correction=torch.zeros_like(log_prob)
        correction[:, 1:]=torch.log(torch.tensor(2.0))
        log_prob += correction

        return action, log_prob.sum(dim=1, keepdim=True)
    
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        
        #影像特徵提取
        self.map_embed=nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 12, 512), 
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.ReLU()
        )
        
        #儀表板特徵提取
        self.dashboard_embed=nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        #動作特徵提取
        self.action_embed=nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.q_head=nn.Sequential(
            nn.Linear(64 + 64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    def forward(self, map_img, dash_vec, action):
        img_feat=self.map_embed(map_img)
        dash_feat=self.dashboard_embed(dash_vec)
        act_feat=self.action_embed(action)
        return self.q_head(torch.cat([img_feat, dash_feat, act_feat], dim=1))

class SACDoubleCritic(nn.Module):
    def __init__(self):
        super(SACDoubleCritic, self).__init__()
        #建立兩個完全獨立的QNetwork
        self.Q1=QNetwork()
        self.Q2=QNetwork()

    def forward(self, map_img, dash_vec, action):
        q1=self.Q1(map_img, dash_vec, action)
        q2=self.Q2(map_img, dash_vec, action)
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self, state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done):
        self.buffer.append((state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done))

    def sample(self, batch_size):
        batch=random.sample(self.buffer, batch_size)
        state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done=zip(*batch)
        return (np.array(state_img), np.array(dash_vec), np.array(action, dtype=np.float32), 
                np.array(reward, dtype=np.float32), np.array(next_state_img), 
                np.array(next_dash_vec), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
class SumTree:
    def __init__(self, capacity):
        self.capacity=capacity
        #樹的節點總數 
        self.tree=np.zeros(2*capacity - 1)
        self.data=np.zeros(capacity, dtype=object)
        self.write=0
        self.n_entries=0

    def _propagate(self, idx, change):
        #更新父節點的值
        parent=(idx - 1) // 2
        self.tree[parent]+=change
        if parent!=0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left=2*idx + 1
        right=left + 1

        if left>=len(self.tree):
            return idx

        if s<=self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self):
        return self.tree[0]

    def add(self, p, data):
        idx=self.write + self.capacity - 1
        self.data[self.write]=data
        self.update(idx, p)
        self.write+=1
        if self.write>=self.capacity:
            self.write=0
        if self.n_entries<self.capacity:
            self.n_entries+=1

    def update(self, idx, p):
        #更新整棵樹
        change=p - self.tree[idx]
        self.tree[idx]=p
        self._propagate(idx, change)

    def get(self, s):
        idx=self._retrieve(0, s)
        data_id=idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_id]  
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.25, beta=0.4, beta_increment=1e-5):
        self.tree=SumTree(capacity)
        self.alpha=alpha  
        self.beta=beta    
        self.beta_increment=beta_increment
        self.epsilon= .01 

    def push(self, state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done):
        #確保新加入的至少會被抽到一次
        max_p=np.max(self.tree.tree[-self.tree.capacity:])
        if max_p==0:
            max_p=1.0
        data=(state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done)
        self.tree.add(max_p, data)

    def sample(self, batch_size):
        idx_batch, data_batch, weights=[], [], []
        
        segment=self.tree.total_priority/batch_size
        self.beta=min(1.0, self.beta + self.beta_increment)

        #防止權重爆炸
        leaf_start=self.tree.capacity - 1
        leaf_end=leaf_start + self.tree.n_entries
        valid_priorities=self.tree.tree[leaf_start:leaf_end]

        p_min=np.min(valid_priorities) / self.tree.total_priority
        p_min=max(p_min, 1e-5)

        max_weight=(p_min*self.tree.n_entries)**(-self.beta)

        for i in range(batch_size):
            x=segment*i
            y=segment*(i + 1)
            s=random.uniform(x, y)
            idx, p, data=self.tree.get(s)
            
            sample_prob=p/self.tree.total_priority
            w=(sample_prob*self.tree.n_entries)**(-self.beta)
            weights.append(w/max_weight)    
            idx_batch.append(idx)
            data_batch.append(data)

        state_img, dash_vec, action, reward, next_state_img, next_dash_vec, done=zip(*data_batch)
        
        return (np.array(state_img), np.array(dash_vec), np.array(action, dtype=np.float32), 
                np.array(reward, dtype=np.float32), np.array(next_state_img), 
                np.array(next_dash_vec), np.array(done, dtype=np.float32),
                np.array(idx_batch), np.array(weights, dtype=np.float32))

    def batch_update(self, tree_idx, abs_errors):
        abs_errors+=self.epsilon
        clipped_errors=np.minimum(abs_errors, 10.0) #防止梯度過大
        ps=np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.n_entries
def preprocess(state_img, dashboard_vec, device):
    #0-1 normalisation for image
    img=torch.FloatTensor(state_img.copy()).permute(2, 0, 1).unsqueeze(0)/255.0
    scales=torch.FloatTensor([100.0, 230.0, 230.0, 230.0, 230.0, 1.0, 1.0, 1.0]).to(device)
    dash=torch.FloatTensor(dashboard_vec).to(device) / scales
    dash=dash.unsqueeze(0) 
    return img.to(device), dash

class SACAgent:
    def __init__(self, device):
        self.device=device
        self.actor=SACActor().to(device)
        self.critic=SACDoubleCritic().to(device)
        self.target_critic=SACDoubleCritic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        #優化
        self.actor_opt=torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    
        #Adaptive Entropy
        self.target_entropy=-3.0 
        self.log_alpha=torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt=torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        self.gamma=0.99
        self.tau=0.005

    def update_parameters(self, buffer, batch_size):
       #Sample batch
        s_img, s_dash, a, r, ns_img, ns_dash, d, tree_idx, is_weights=buffer.sample(batch_size)        
        
        #Setup tensors
        s_img=torch.FloatTensor(s_img).permute(0, 3, 1, 2).to(self.device)/255.0
        ns_img=torch.FloatTensor(ns_img).permute(0, 3, 1, 2).to(self.device)/255.0

        scales=torch.FloatTensor([100.0, 230.0, 230.0, 230.0, 230.0, 1.0, 1.0, 1.0]).to(self.device)
        s_dash, ns_dash=torch.FloatTensor(s_dash).to(self.device)/scales, torch.FloatTensor(ns_dash).to(self.device)/scales
        a, r, d=torch.FloatTensor(a).to(self.device), torch.FloatTensor(r).unsqueeze(1).to(self.device), torch.FloatTensor(d).unsqueeze(1).to(self.device)
        weights=torch.FloatTensor(is_weights).unsqueeze(1).to(self.device) 

        with torch.no_grad():
            next_action, next_log_prob=self.actor.sample(ns_img, ns_dash)
            q1_t, q2_t=self.target_critic(ns_img, ns_dash, next_action)
            target_v=torch.min(q1_t, q2_t) - self.log_alpha.exp()*next_log_prob
            target_q=r + (1 - d)*self.gamma*target_v

        #Critic update with PER weights
        curr_q1, curr_q2=self.critic(s_img, s_dash, a)
        critic_loss1=(weights*F.mse_loss(curr_q1, target_q, reduction='none')).mean()
        critic_loss2=(weights*F.mse_loss(curr_q2, target_q, reduction='none')).mean()
        critic_loss=critic_loss1 + critic_loss2
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        #Update PER priorities using TD-error
        with torch.no_grad():
            td_error=(torch.abs(curr_q1 - target_q) + torch.abs(curr_q2 - target_q))/2.0
            td_error=td_error.cpu().numpy().flatten()
        buffer.batch_update(tree_idx, td_error)

        #Actor update
        new_a, log_prob=self.actor.sample(s_img, s_dash)
        q1_new, q2_new=self.critic(s_img, s_dash, new_a)
        actor_loss=(self.log_alpha.exp()*log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        #Alpha update
        alpha_loss=-(self.log_alpha*(log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        #Soft Update
        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.copy_(t.data*(1.0 - self.tau) + s.data*self.tau)

    def save_checkpoint(self, checkpoint_path, episode, now_score, best_score):
        checkpoint = {
            'episode': episode,
            'now_score': now_score,
            'best_score': best_score,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'target_critic_state': self.target_critic.state_dict(),
            'actor_opt_state': self.actor_opt.state_dict(),
            'critic_opt_state': self.critic_opt.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_opt_state': self.alpha_opt.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint=torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'])
        self.alpha_opt.load_state_dict(checkpoint['alpha_opt_state'])
        return checkpoint['episode'], checkpoint['now_score'], checkpoint['best_score']

def save_buffer(buffer, path):
    with open(path, 'wb') as f:
        state={
            'tree': buffer.tree.tree,
            'data': buffer.tree.data,
            'write': buffer.tree.write,
            'n_entries': buffer.tree.n_entries,
            'alpha': buffer.alpha,
            'beta': buffer.beta,
            'beta_increment': buffer.beta_increment,
            'epsilon': buffer.epsilon
        }
        pickle.dump(state, f)

def load_buffer(buffer, path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            state=pickle.load(f)
        
        buffer.tree.tree=state['tree']
        buffer.tree.data=state['data']
        buffer.tree.write=state['write']
        buffer.tree.n_entries=state['n_entries']
    
        buffer.alpha=state.get('alpha', buffer.alpha)
        buffer.beta=state.get('beta', buffer.beta)
        buffer.beta_increment=state.get('beta_increment', buffer.beta_increment)
        buffer.epsilon=state.get('epsilon', buffer.epsilon)
        return True

    else:
        return False

def evaluate_agent(agent, seed, trials):
    env_eval=gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=True, max_episode_steps=5000, lap_complete_percent=0.99)
    env_eval.action_space.seed(seed)
    env_eval.observation_space.seed(seed)
    env_eval.reset(seed=seed)

    rewards=[]

    for ep in range(trials):
        state, info=env_eval.reset()
        for t in range(50):
            env_eval.step(np.array([0, 0, 0]))
        state=state[2:82]
        dashboard=np.zeros(8)
        tile_hit=0
        
        for t in range(num_episode_step):
            img_t, dash_t=preprocess(state, dashboard, device)
            with torch.no_grad():
                action, x=agent.actor.sample(img_t, dash_t)
            action=action.cpu().numpy()[0]

            next_state, reward, done, truncated, info=env_eval.step(action)
            next_state=next_state[2:82]
            
            
            car=env_eval.unwrapped.car
            next_dashboard=np.concatenate((np.array([np.linalg.norm(car.hull.linearVelocity)] + [w.omega for w in car.wheels]), action))
            
            state, dashboard=next_state, next_dashboard
            if (reward>0):
              tile_hit+=1
        
        rewards.append(tile_hit)
    
    rewards=np.array(rewards)
    return rewards.mean(), rewards.max(), rewards.min()


env=gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=True, max_episode_steps=5000, lap_complete_percent=0.99)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
env.reset(seed=SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_buffer_size=500000
min_buffer_size=10000
num_total_steps=1000000
num_episode_step=5000
batch_size=128

buffer=PrioritizedReplayBuffer(max_buffer_size)
agent=SACAgent(device)

checkpoint_path="sac_car_checkpoint.pth"
buffer_path=f"sac_car_buffer.pkl"
total_steps=0
now_score=0
best_score=float('-inf')
mean_rewards=[]
max_rewards=[]
min_rewards=[]

if (os.path.exists(checkpoint_path)):
    load_buffer(buffer, buffer_path)
    total_steps, now_score, best_score=agent.load_checkpoint(checkpoint_path)
    total_steps+=1

for episode in range(total_steps):
    state, info=env.reset()

while (True):
    state, info=env.reset()
    for t in range(50):
        env.step(np.array([0, 0, 0]))
    state=state[2:82]
    dashboard=np.zeros(8)
    episode_reward=0
    stay_neg=0
    
    for t in range(num_episode_step):
       
        img_t, dash_t=preprocess(state, dashboard, device)
        with torch.no_grad():
            action, _ =agent.actor.sample(img_t, dash_t)
        action=action.cpu().numpy()[0]

        next_state, reward, done, truncated, info=env.step(action)
        next_state=next_state[2:82]
        
        car=env.unwrapped.car
        next_dashboard=np.concatenate((np.array([np.linalg.norm(car.hull.linearVelocity)] + [w.omega for w in car.wheels]), action))
        
        #判斷卡死
        if (reward<0):
          stay_neg+=1
        else:
          stay_neg=0
        if (stay_neg>500):
          done=True
        
        #判斷飛出去
        on_grass=True
        for wheel in env.unwrapped.car.wheels:
            if len(wheel.tiles)>0:
                on_grass=False
                break
        if on_grass:
          done=True
        
        #存入 Buffer
        buffer.push(state, dashboard, action, reward, next_state, next_dashboard, done)
        
        state, dashboard=next_state, next_dashboard
        episode_reward+=reward
    
        total_steps+=1
        
        # 訓練模型
        if len(buffer)>=min_buffer_size:
            if (total_steps%1000==0):
                print(f"total_steps: {total_steps}")
    
                now_mean, now_max, now_min=evaluate_agent(agent, SEED+100, 5)
                best_score=max(best_score, now_mean)
    
                mean_rewards.append(now_mean)
                max_rewards.append(now_max)
                min_rewards.append(now_min)
    
                df=pd.DataFrame({
                    'mean': mean_rewards,
                    'max': max_rewards,
                    'min': min_rewards
                })
                df.to_csv('training_cruve.csv', index=False, encoding='utf-8-sig')
                agent.save_checkpoint(checkpoint_path, total_steps, now_mean, best_score)
                save_buffer(buffer, buffer_path)
                
                if (now_mean==best_score):
                    os.system(f"cp {checkpoint_path} sac_car_best_model.pth")

            agent.update_parameters(buffer, batch_size)

        if (total_steps==num_total_steps):
            break

        if done or truncated:
            break

    if (total_steps==num_total_steps):
        break
