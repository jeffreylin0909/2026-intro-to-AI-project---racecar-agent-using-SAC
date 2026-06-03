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

from SAC_class import PrioritizedReplayBuffer, SACAgent
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed=42):
    # Python 內建亂數
    random.seed(seed)
    # Numpy 亂數
    np.random.seed(seed)
    # PyTorch 亂數
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 若有多張 GPU
    
    # 針對卷積運算的決定性設定 (注意：這可能會稍微降低運算效能)
    # force gpu to be deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"已設定亂數種子為: {seed}")

# 呼叫設定
SEED = 67
set_seed(SEED)

def preprocess(state_img, dashboard_vec, device):
    # 影像: 標準 0~1 歸一化
    img = torch.FloatTensor(state_img.copy()).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # 建立一個與 dashboard_vec 同形狀的 scale 向量
    # 速度上限 100, 輪速上限 230
    # normalize to [0,1] for each element
    scales = torch.FloatTensor([100.0, 230.0, 230.0, 230.0, 230.0, 1.0, 1.0, 1.0]).to(device)
    
    # 轉換為 Tensor 並直接除以各自的上限
    dash = torch.FloatTensor(dashboard_vec).to(device) / scales
    dash = dash.unsqueeze(0) # 補上 Batch 維度
    
    return img.to(device), dash

def save_buffer(buffer, path):
    with open(path, 'wb') as f:
        state = {
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
            state = pickle.load(f)
        
        # 還原 SumTree 內部狀態
        buffer.tree.tree = state['tree']
        buffer.tree.data = state['data']
        buffer.tree.write = state['write']
        buffer.tree.n_entries = state['n_entries']
        
        # 還原超參數狀態
        buffer.alpha = state.get('alpha', buffer.alpha)
        buffer.beta = state.get('beta', buffer.beta)
        buffer.beta_increment = state.get('beta_increment', buffer.beta_increment)
        buffer.epsilon = state.get('epsilon', buffer.epsilon)
        
        return True
    else:
        return False

def evaluate_agent(agent, seed, num_trials):
    env_eval = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=True, max_episode_steps=5000, lap_complete_percent=0.99)
    env_eval.action_space.seed(seed)
    env_eval.observation_space.seed(seed)
    env_eval.reset(seed=seed)

    rewards = []

    for ep in range(num_trials):
        state, info = env_eval.reset()
        for t in range(50):
            env_eval.step(np.array([0, 0, 0]))
        state = state[2:82]
        dashboard = np.zeros(8)
        tile_hit = 0
        
        for t in range(num_episode_step):
            # 選擇動作
            img_t, dash_t = preprocess(state, dashboard, device)
            with torch.no_grad():
                action, _ = agent.actor.sample(img_t, dash_t)
            action = action.cpu().numpy()[0]

            # 執行動作
            next_state, reward, done, truncated, info = env_eval.step(action)
            next_state = next_state[2:82]
            
            # 更新物理量 (Dashboard)
            car = env_eval.unwrapped.car
            next_dashboard = np.concatenate((np.array([np.linalg.norm(car.hull.linearVelocity)] + [w.omega for w in car.wheels]), action))
            
            state, dashboard = next_state, next_dashboard
            if (reward>0):
              tile_hit += 1
        
        rewards.append(tile_hit)
    
    rewards = np.array(rewards)
    return rewards.mean(), rewards.max(), rewards.min()

if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=True, max_episode_steps=5000, lap_complete_percent=0.99)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env.reset(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_buffer_size = 500000
    min_buffer_size = 10000
    num_total_steps = 1000000
    num_episode_step = 5000
    batch_size = 128

    buffer = PrioritizedReplayBuffer(max_buffer_size)
    agent = SACAgent(device)

    checkpoint_path = "sac_car_checkpoint.pth"
    buffer_path = f"sac_car_buffer.pkl"
    total_steps = 0
    now_score = 0
    best_score = float('-inf')
    mean_rewards = []
    max_rewards = []
    min_rewards = []

    if (os.path.exists(checkpoint_path)):
        load_buffer(buffer, buffer_path)
        total_steps, now_score, best_score = agent.load_checkpoint(checkpoint_path)
        total_steps += 1

        print(f"Resuming from checkpoint...\n \
              step = {total_steps}\n\
              last_score = {now_score:.1f}\n\
              best_score = {best_score:.1f}\n\
              buffer_size = {len(buffer)}\n")
    else:
        print("No saved checkpoint found, start from beginning...")

    for episode in range(total_steps):
        state, info = env.reset()


    while (True):
        state, info = env.reset()
        for t in range(50):
            env.step(np.array([0, 0, 0]))
        state = state[2:82]
        dashboard = np.zeros(8)
        episode_reward = 0
        stay_neg = 0
    
        for t in range(num_episode_step):
            # 選擇動作
            img_t, dash_t = preprocess(state, dashboard, device)
            with torch.no_grad():
                action, _ = agent.actor.sample(img_t, dash_t)
            action = action.cpu().numpy()[0]

            # 執行動作
            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state[2:82]
            
            # 更新物理量 (Dashboard)
            car = env.unwrapped.car
            next_dashboard = np.concatenate((np.array([np.linalg.norm(car.hull.linearVelocity)] + [w.omega for w in car.wheels]), action))
            
            # 判斷卡死
            if (reward<0):
                stay_neg += 1
            else:
                stay_neg = 0

            if (stay_neg>500):
                print(f"[Step {t}] Early stop: negative steps")
                done = True
            
            # 判斷飛出去
            on_grass = True
            for wheel in env.unwrapped.car.wheels:
                if len(wheel.tiles) > 0:
                    on_grass = False
                    break
            if on_grass:
                print(f"[Step {t}] Early stop: car on grass")
                done = True
            
            # 存入 Buffer
            buffer.push(state, dashboard, action, reward, next_state, next_dashboard, done)
            
            state, dashboard = next_state, next_dashboard
            episode_reward += reward
        
            total_steps += 1
            
            # 訓練模型
            if len(buffer) >= min_buffer_size:
            
                if (total_steps%1000 == 0):
                    now_mean, now_max, now_min = evaluate_agent(agent, SEED+100, 5)
                    best_score = max(best_score, now_mean)

                    print(f"Total steps: {total_steps}, now_mean={now_mean:.1f}  max={now_max:.1f}  min={now_min:.1f}  best={best_score:.1f}")

                    mean_rewards.append(now_mean)
                    max_rewards.append(now_max)
                    min_rewards.append(now_min)
        
                    df = pd.DataFrame({
                        'mean': mean_rewards,
                        'max': max_rewards,
                        'min': min_rewards
                    })
        
                    df.to_csv('training_curve.csv', index=False, encoding='utf-8-sig')
        
                    agent.save_checkpoint(checkpoint_path, total_steps, now_mean, best_score)
                    save_buffer(buffer, buffer_path)
                    
                    if (now_mean==best_score):
                        print(f"New best model saved")
                        os.system(f"cp {checkpoint_path} sac_car_best_model.pth")

                agent.update_parameters(buffer, batch_size)

            if (total_steps == num_total_steps) or done or truncated:
                break

            #if done or truncated:
            #    break

        if (total_steps == num_total_steps):
            print(f"Training complete: {total_steps} steps")
            break

