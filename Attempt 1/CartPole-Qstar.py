from logging import warn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
import time

class CartPoleQLearningAgent:
    def __init__(
        self,
        n_bins: int = 10,  # Number of bins for discretization
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        # CartPole has 4 continuous state variables and 2 actions
        self.n_bins = n_bins
        self.q_table = {}  # Using dictionary for sparse state space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = 2  # CartPole has 2 actions (left/right)
        
        # State space bounds for discretization
        self.bins = {
            'position': np.linspace(-2.4, 2.4, n_bins),
            'velocity': np.linspace(-4, 4, n_bins),
            'angle': np.linspace(-0.418, 0.418, n_bins),
            'angular_velocity': np.linspace(-4, 4, n_bins)
        }

    def discretize_state(self, state) -> tuple:
        """Convert continuous state to discrete state"""
        position, velocity, angle, angular_velocity = state
        
        discrete_state = (
            np.digitize(position, self.bins['position']),
            np.digitize(velocity, self.bins['velocity']),
            np.digitize(angle, self.bins['angle']),
            np.digitize(angular_velocity, self.bins['angular_velocity'])
        )
        return discrete_state

    def choose_action(self, state: tuple) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        
        # Initialize state value if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
            
        return int(np.argmax(self.q_table[state]))

    def learn(
        self, state: tuple, action: int, reward: float, next_state: tuple, done: bool
    ) -> None:
        """Update Q-value for state-action pair"""
        # Initialize states if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        # Q-learning update
        next_max = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_and_evaluate(
    episodes: int = 3000,
    eval_interval: int = 300,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the agent on CartPole
    """
    env = gym.make('CartPole-v1')
    agent = CartPoleQLearningAgent()

    rewards_history = []
    steps_history = []
    success_history = []  # Episodes where reward > 195 (CartPole success threshold)

    for episode in range(episodes):
        state, _ = env.reset()
        state = agent.discretize_state(state)
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        while not (done or truncated):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = agent.discretize_state(next_state)
            
            # Modify reward to encourage learning
            modified_reward = reward
            if done and steps < 195:  # If failed before success threshold
                modified_reward = -10
            elif steps >= 195:  # If reached success threshold
                modified_reward = 10

            agent.learn(state, action, modified_reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

        rewards_history.append(episode_reward)
        steps_history.append(steps)
        success_history.append(1 if episode_reward >= 195 else 0)  # Success if episode lasts 195 steps

        if episode % eval_interval == 0:
            success_rate = np.mean(success_history[-eval_interval:]) if success_history else 0
            avg_reward = np.mean(rewards_history[-eval_interval:])
            print(f"Episode {episode}/{episodes} | Avg Reward: {avg_reward:.1f} | Success Rate: {success_rate:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history, steps_history, success_history

def plot_metrics(rewards: List[float], steps: List[float], success: List[float], 
                window_size: int = 100) -> None:
    """
    Plot training metrics with smoothing
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    episodes = range(len(rewards))

    def smooth(data, window_size):
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')

    # Plot smoothed rewards
    smoothed_rewards = smooth(rewards, window_size)
    ax1.plot(episodes[window_size-1:], smoothed_rewards, label='Q-Learning', color='blue')
    ax1.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
    ax1.set_title('Average Reward per Episode (Smoothed)', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()

    # Plot smoothed steps
    smoothed_steps = smooth(steps, window_size)
    ax2.plot(episodes[window_size-1:], smoothed_steps, label='Q-Learning', color='blue')
    ax2.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
    ax2.set_title('Steps per Episode (Smoothed)', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    ax2.legend()

    # Plot smoothed success rate
    smoothed_success = smooth(success, window_size)
    ax3.plot(episodes[window_size-1:], smoothed_success, label='Q-Learning', color='blue')
    ax3.set_title('Success Rate (Smoothed)', fontsize=12)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run training
    rewards, steps, success = train_and_evaluate(episodes=3000)
    
    # Plot results
    plot_metrics(rewards, steps, success)