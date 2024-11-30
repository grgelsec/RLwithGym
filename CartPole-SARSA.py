from logging import warn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
import time
from tqdm import tqdm

class CartPoleSARSAAgent:
    def __init__(
        self,
        n_bins: int = 8,  # Reduced bins for simpler state space
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
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
        
        # Simplified bin ranges focusing on critical regions
        self.bins = {
            'position': np.linspace(-2.4, 2.4, n_bins),
            'velocity': np.linspace(-2, 2, n_bins),
            'angle': np.linspace(-0.2, 0.2, n_bins),
            'angular_velocity': np.linspace(-2, 2, n_bins)
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
        self, state: tuple, action: int, reward: float, next_state: tuple, next_action: int, done: bool
    ) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        next_q = 0 if done else self.q_table[next_state][next_action]
        target = reward + self.gamma * next_q
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_single_trial(trial_num: int, episodes: int = 2000) -> Tuple[List[float], List[float], List[float]]:
    """Run a single training trial with improved reward structure"""
    env = gym.make('CartPole-v1')
    agent = CartPoleSARSAAgent()

    rewards_history = []
    steps_history = []
    success_history = []

    pbar = tqdm(range(episodes), desc=f'Trial {trial_num+1}/8', leave=True)
    
    for episode in pbar:
        state, _ = env.reset()
        state = agent.discretize_state(state)
        action = agent.choose_action(state)
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = agent.discretize_state(next_state)
            next_action = agent.choose_action(next_state)
            
            # Simple reward structure focusing on survival
            if done and steps < 195:
                modified_reward = -1
            else:
                modified_reward = reward  # Keep original reward

            agent.learn(state, action, modified_reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            episode_reward += reward
            steps += 1

            if steps >= 500:  # Cap episode length
                truncated = True

        rewards_history.append(episode_reward)
        steps_history.append(steps)
        success_history.append(1 if steps >= 195 else 0)

        if episode % 100 == 0:
            success_rate = np.mean(success_history[-100:]) if success_history else 0
            avg_steps = np.mean(steps_history[-100:]) if steps_history else 0
            pbar.set_postfix({
                'Success Rate': f'{success_rate:.2%}', 
                'Epsilon': f'{agent.epsilon:.2f}',
                'Avg Steps': f'{avg_steps:.1f}'
            })

    env.close()
    return rewards_history, steps_history, success_history

def run_multiple_trials(num_trials: int = 8, episodes: int = 2000) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Run multiple training trials and return all results"""
    all_rewards = []
    all_steps = []
    all_success = []
    
    print(f"\nStarting SARSA training for {num_trials} trials, {episodes} episodes each...")
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
        ax.plot(x, mean, label='Mean across 8 trials', color='green', linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color='green', alpha=0.2, label='±1 STD')
        
        if title in ['Average Reward', 'Steps']:
            ax.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
        
        ax.set_title(f'{title} over Episodes (Smoothed)', fontsize=12)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
        if title == 'Success Rate':
            ax.set_ylim(-0.1, 1.1)

    plt.suptitle('CartPole SARSA Performance over 8 Trials', fontsize=14)
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
    
    print("\nPer-Trial Performance:")
    for i, (reward, success_rate) in enumerate(zip(final_rewards, final_success_rates), 1):
        print(f"Trial {i}: Success Rate = {success_rate:.2%}, Average Reward = {reward:.1f}")
    
    best_trial = np.argmax(final_success_rates)
    print(f"\nBest Trial ({best_trial + 1}):")
    print(f"Final Success Rate: {final_success_rates[best_trial]:.2%}")
    print(f"Final Average Reward: {final_rewards[best_trial]:.1f}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    all_rewards, all_steps, all_success = run_multiple_trials(num_trials=8, episodes=2000)
    
    plot_multi_trial_metrics(all_rewards, all_steps, all_success)
    
    print_trial_statistics(all_rewards, all_steps, all_success)