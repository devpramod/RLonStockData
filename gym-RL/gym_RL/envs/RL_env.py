import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import json
import pandas as pd
import numpy as np

class RLEnv(gym.Env):
    """Custom environment for stock market data"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, history_t=90):
#         super(RL_env, self).__init__()
        self.data = data
        self.history_t = history_t
        
        self.reward_range = (-1, 1) 
        
        # Actions of the format Buy x%, Sell x%, Hold, etc.
#         self.action_space = spaces.Box(
#         low=np.array([0]), high=np.array([2]), dtype=np.float16)
        self.action_space = spaces.Discrete(3)
        
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
        low=0, high=1, shape=(91,), dtype=np.float16)

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return np.array([self.position_value] + self.history) # obs
    
    def step(self, act):
        reward = 0
        
        # act = 0: stay, 1: buy, 2: sell
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2: # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []
        
        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])
        
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        return np.asarray([self.position_value] + self.history), np.asarray(reward), self.done # obs, reward, done 
