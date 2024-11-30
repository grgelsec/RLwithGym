from logging import warn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
import time

class FastQLearningAgent:
    def __init__(
        self,
        state_space: int,
        action_space: int,
        learning_rate: float = 0.2,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,  # Slightly adjusted for longer training
    ):
        self.q_table = np.zeros((state_space, action_space))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space

    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        return int(np.argmax(self.q_table[state]))

    def learn(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        next_max = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def fast_train_and_evaluate(
    env_name: str,
    episodes: int = 3000,  # Increased to 3000 episodes
    eval_interval: int = 300,  # Checking eval every 300 episodes
) -> Tuple[List[float], List[float], List[float]]:
    """
    Faster training implementation with periodic evaluation
    """
    env = gym.make(env_name, render_mode=None)
    eval_env = gym.make(env_name, render_mode=None)

    state_space = env.observation_space.n
    action_space = env.action_space.n
    agent = FastQLearningAgent(
        state_space=state_space,
        action_space=action_space
    )

    rewards_history = []
    steps_history = []
    success_history = []

    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        while not (done or truncated):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Modified reward structure for faster learning
            modified_reward = reward
            if done and reward == 0:
                modified_reward = -2
            elif done and reward == 1:
                modified_reward = 5

            agent.learn(state, action, modified_reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

        rewards_history.append(episode_reward)
        steps_history.append(steps)
        success_history.append(1 if episode_reward > 0 else 0)

        # Print progress every eval_interval episodes
        if episode % eval_interval == 0:
            success_rate = np.mean(success_history[-eval_interval:]) if success_history else 0
            print(f"Episode {episode}/{episodes} | Success Rate: {success_rate:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    eval_env.close()
    return rewards_history, steps_history, success_history

def plot_comparison_metrics(q_rewards: List[float], q_steps: List[float], q_success: List[float], 
                          window_size: int = 100) -> None:  # Increased window size for smoother plots
    """
    Plot training metrics with smoothing
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))  # Increased figure size
    episodes = range(len(q_rewards))

    # Smoothing function
    def smooth(data, window_size):
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')

    # Plot smoothed rewards
    smoothed_rewards = smooth(q_rewards, window_size)
    ax1.plot(episodes[window_size-1:], smoothed_rewards, label='Q-Learning', color='blue')
    ax1.set_title('Average Reward per Episode (Smoothed)', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()

    # Plot smoothed steps
    smoothed_steps = smooth(q_steps, window_size)
    ax2.plot(episodes[window_size-1:], smoothed_steps, label='Q-Learning', color='blue')
    ax2.set_title('Average Steps per Episode (Smoothed)', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    ax2.legend()

    # Plot smoothed success rate
    smoothed_success = smooth(q_success, window_size)
    ax3.plot(episodes[window_size-1:], smoothed_success, label='Q-Learning', color='blue')
    ax3.set_title('Success Rate (Smoothed)', fontsize=12)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    # Run training
    rewards, steps, success = fast_train_and_evaluate("FrozenLake-v1", episodes=3000)
    
    # Plot results
    plot_comparison_metrics(rewards, steps, success)