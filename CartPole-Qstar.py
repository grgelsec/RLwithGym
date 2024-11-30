from logging import warn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
import time
from tqdm import tqdm

class CartPoleQLearningAgent:
    def __init__(
        self,
        n_bins: int = 10,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_bins = n_bins
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = 2
        
        # State space bins for discretization
        self.bins = {
            'position': np.linspace(-2.4, 2.4, n_bins),
            'velocity': np.linspace(-4, 4, n_bins),
            'angle': np.linspace(-0.418, 0.418, n_bins),
            'angular_velocity': np.linspace(-4, 4, n_bins)
        }

    def discretize_state(self, state) -> tuple:
        position, velocity, angle, angular_velocity = state
        discrete_state = (
            np.digitize(position, self.bins['position']),
            np.digitize(velocity, self.bins['velocity']),
            np.digitize(angle, self.bins['angle']),
            np.digitize(angular_velocity, self.bins['angular_velocity'])
        )
        return discrete_state

    def choose_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def learn(
        self, state: tuple, action: int, reward: float, next_state: tuple, done: bool
    ) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        next_max = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_single_trial(trial_num: int, episodes: int = 2000) -> Tuple[List[float], List[float], List[float]]:
    """Run a single training trial"""
    env = gym.make('CartPole-v1')
    agent = CartPoleQLearningAgent()

    rewards_history = []
    steps_history = []
    success_history = []

    # Progress bar for each trial
    pbar = tqdm(range(episodes), desc=f'Trial {trial_num+1}/7', leave=True)
    
    for episode in pbar:
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
            
            modified_reward = reward
            if done and steps < 195:
                modified_reward = -10
            elif steps >= 195:
                modified_reward = 10

            agent.learn(state, action, modified_reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

        rewards_history.append(episode_reward)
        steps_history.append(steps)
        success_history.append(1 if episode_reward >= 195 else 0)

        # Update progress bar with current metrics
        if episode % 100 == 0:
            recent_success_rate = np.mean(success_history[-100:]) if success_history else 0
            pbar.set_postfix({'Success Rate': f'{recent_success_rate:.2%}'})

    env.close()
    return rewards_history, steps_history, success_history

def run_multiple_trials(num_trials: int = 7, episodes: int = 2000) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Run multiple training trials and return all results"""
    all_rewards = []
    all_steps = []
    all_success = []
    
    print("\nStarting training for 7 trials, 2000 episodes each...")
    for trial in range(num_trials):
        rewards, steps, success = run_single_trial(trial, episodes)
        all_rewards.append(rewards)
        all_steps.append(steps)
        all_success.append(success)
        
    return all_rewards, all_steps, all_success

def plot_multi_trial_metrics(all_rewards: List[List[float]], all_steps: List[List[float]], 
                           all_success: List[List[float]], window_size: int = 50) -> None:
    """Plot metrics for multiple trials with confidence intervals"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    episodes = range(len(all_rewards[0]))

    def smooth_data(data_list):
        smoothed_data = []
        for trial_data in data_list:
            kernel = np.ones(window_size) / window_size
            smoothed_trial = np.convolve(trial_data, kernel, mode='valid')
            smoothed_data.append(smoothed_trial)
        return np.array(smoothed_data)

    # Calculate statistics for each metric
    for ax, data, title, ylabel in zip(
        [ax1, ax2, ax3],
        [all_rewards, all_steps, all_success],
        ['Average Reward', 'Steps', 'Success Rate'],
        ['Reward', 'Steps', 'Success Rate']
    ):
        smoothed_data = smooth_data(data)
        mean = np.mean(smoothed_data, axis=0)
        std = np.std(smoothed_data, axis=0)
        
        x = range(window_size-1, len(data[0]))
        ax.plot(x, mean, label='Mean across 7 trials', color='blue', linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color='blue', alpha=0.2, label='±1 STD')
        
        if title in ['Average Reward', 'Steps']:
            ax.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
        
        ax.set_title(f'{title} over Episodes (Smoothed)', fontsize=12)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
        if title == 'Success Rate':
            ax.set_ylim(-0.1, 1.1)

    plt.suptitle('CartPole Q-Learning Performance over 7 Trials', fontsize=14)
    plt.tight_layout()
    plt.show()

def print_trial_statistics(all_rewards: List[List[float]], all_steps: List[List[float]], 
                         all_success: List[List[float]]) -> None:
    """Print summary statistics for all trials"""
    final_rewards = [np.mean(rewards[-100:]) for rewards in all_rewards]
    final_success_rates = [np.mean(success[-100:]) for success in all_success]
    
    print("\nFinal Statistics (last 100 episodes of each trial):")
    print(f"Average Reward: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"Success Rate: {np.mean(final_success_rates):.2%} ± {np.std(final_success_rates):.2%}")
    
    # Statistics for each trial
    print("\nPer-Trial Performance:")
    for i, (reward, success_rate) in enumerate(zip(final_rewards, final_success_rates), 1):
        print(f"Trial {i}: Success Rate = {success_rate:.2%}, Average Reward = {reward:.1f}")
    
    # Best trial
    best_trial = np.argmax(final_success_rates)
    print(f"\nBest Trial ({best_trial + 1}):")
    print(f"Final Success Rate: {final_success_rates[best_trial]:.2%}")
    print(f"Final Average Reward: {final_rewards[best_trial]:.1f}")

if __name__ == "__main__":
    # Run 7 trials with 2000 episodes each
    all_rewards, all_steps, all_success = run_multiple_trials(num_trials=7, episodes=2000)
    
    # Plot results
    plot_multi_trial_metrics(all_rewards, all_steps, all_success)
    
    # Print statistics
    print_trial_statistics(all_rewards, all_steps, all_success)