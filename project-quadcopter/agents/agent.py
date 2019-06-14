from keras import layers, models, optimizers
from keras import backend as K

from .actor import Actor
from .critic import Critic
import numpy as np
import random
from collections import namedtuple, deque


class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience', 
                                     field_names=['state', 'action', 
                                                  'reward', 'next_state', 
                                                  'done'])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self):
        return len(self.buffer)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class UserAgent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_dim = task.state_size
        self.action_dim = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Actor
        self.actor_local = Actor(self.state_dim, self.action_dim, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.action_low, self.action_high)
        
        # Critic
        self.critic_local = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        
         # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.25
        self.noise = OUNoise(self.action_dim, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayMemory(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        
        self.total_reward = 0
        self.count = 0
        self.score = 0
        
        self.best_score = -np.inf
        self.best_w = None
        
        print(">>> agent param")
        print(">>> gamma {} tau {}".format(self.gamma, self.tau))
        print(">>> theta {} sigma {}".format(self.exploration_theta, self.exploration_sigma))

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        
        self.total_reward = 0
        self.count = 0
        self.score = 0
        return state

    def step(self, action, reward, next_state, done):
        self.total_reward += reward
        self.count += 1
        
        self.score = self.total_reward
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = [self.actor_local.model.get_weights(), self.critic_local.model.get_weights()]
#         elif self.best_w is not None:
#             self.actor_local.model.set_weights(self.best_w[0])
#             self.critic_local.model.set_weights(self.best_w[1])
        
        self.memory.add(self.last_state, action, reward, next_state, done)
        if (len(self.memory) > self.batch_size):
            experience = self.memory.sample()
            self.learn(experience)
            
        self.last_state = next_state
    
    # 进行一次动作的预测
    def act(self, state):
        state = np.reshape(state, [-1, self.state_dim])
        action = self.actor_local.model.predict(state)[0]
        return list(action+self.noise.sample())

    def learn(self, experience):
        states = np.vstack([e.state for e in experience if e is not None])
        actions = np.array([e.action for e in experience if e is not None]).astype(np.float).reshape(-1, self.action_dim)
        rewards = np.array([e.reward for e in experience if e is not None]).astype(np.float).reshape(-1, 1)
        dones = np.array([e.done for e in experience if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_state = np.vstack([e.next_state for e in experience if e is not None])
        
        # 计算Q的目标值，使用贝尔曼公式，因此需要知道下一状态的动作值
        next_action = self.actor_target.model.predict_on_batch(next_state)
        Q_next = self.critic_target.model.predict_on_batch([next_state, next_action])
        
        # 如果下一次是done状态的话，不计入动作值中
        Q_current_label = rewards + self.gamma*Q_next*(1-dones)
        
        # 训练critic
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_current_label)
        
        # 训练actor
        action_grad = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_dim))
        self.actor_local.train_fn([states, action_grad, 1])
        
        # 更新权重
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model) 
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
