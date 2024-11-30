from logging import warn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
import time
from tqdm import tqdm

class FrozenLakeSARSAAgent:
    def __init__(
        self,
        state_space: int,
        action_space: int,
        learning_rate: float = 0.2,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
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
        self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool
    ) -> None:
        # SARSA update rule: uses the actual next action instead of the maximum Q-value
        next_q = 0 if done else self.q_table[next_state, next_action]
        target = reward + self.gamma * next_q
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def visualize_policy(q_table):
    """Visualize the learned policy as arrows"""
    actions = ['←', '↓', '→', '↑']
    policy = np.argmax(q_table, axis=1)
    
    print("\nLearned Policy:")
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(actions[policy[state]], end=' ')
        print()
    print()

def run_single_trial(trial_num: int, episodes: int = 1000) -> Tuple[List[float], List[float], List[float]]:
    """Run a single training trial with SARSA"""
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    state_space = env.observation_space.n
    action_space = env.action_space.n
    agent = FrozenLakeSARSAAgent(
        state_space=state_space,
        action_space=action_space
    )

    rewards_history = []
    steps_history = []
    success_history = []

    pbar = tqdm(range(episodes), desc=f'Trial {trial_num+1}/10', leave=True)
    
    for episode in pbar:
        state, _ = env.reset()
        action = agent.choose_action(state)  # Choose initial action
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)  # Choose next action
            
            # Enhanced reward structure
            modified_reward = reward
            if done and reward == 0:  # Fell in a hole
                modified_reward = -5
            elif done and reward == 1:  # Reached goal
                modified_reward = 10
            elif not done:  # Small negative reward for each step
                modified_reward = -0.1

            agent.learn(state, action, modified_reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            episode_reward += reward
            steps += 1

            if steps > 100:  # Prevent very long episodes
                done = True

        rewards_history.append(episode_reward)
        steps_history.append(steps)
        success_history.append(1 if episode_reward > 0 else 0)

        # Show policy every 200 episodes for the first trial
        if trial_num == 0 and episode % 200 == 0 and episode > 0:
            print(f"\nEpisode {episode} Policy:")
            visualize_policy(agent.q_table)

        if episode % 50 == 0:  # More frequent updates
            recent_success_rate = np.mean(success_history[-50:]) if success_history else 0
            pbar.set_postfix({'Success Rate': f'{recent_success_rate:.2%}', 'Epsilon': f'{agent.epsilon:.2f}'})

    env.close()
    return rewards_history, steps_history, success_history

def run_multiple_trials(num_trials: int = 10, episodes: int = 1000) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
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
                           all_success: List[List[float]], window_size: int = 30) -> None:
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
        ax.plot(x, mean, label='Mean across 10 trials', color='green', linewidth=2)  # Changed color to differentiate from Q-learning
        ax.fill_between(x, mean - std, mean + std, color='green', alpha=0.2, label='±1 STD')
        
        ax.set_title(f'{title} over Episodes (Smoothed)', fontsize=12)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
        if title == 'Success Rate':
            ax.set_ylim(-0.1, 1.1)

    plt.suptitle('FrozenLake SARSA Performance over 10 Trials', fontsize=14)
    plt.tight_layout()
    plt.show()

def print_trial_statistics(all_rewards: List[List[float]], all_steps: List[List[float]], 
                         all_success: List[List[float]]) -> None:
    """Print summary statistics for all trials"""
    final_rewards = [np.mean(rewards[-50:]) for rewards in all_rewards]
    final_success_rates = [np.mean(success[-50:]) for success in all_success]
    
    print("\nFinal Statistics (last 50 episodes of each trial):")
    print(f"Average Reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    print(f"Success Rate: {np.mean(final_success_rates):.2%} ± {np.std(final_success_rates):.2%}")
    
    # Statistics for each trial
    print("\nPer-Trial Performance:")
    for i, (reward, success_rate) in enumerate(zip(final_rewards, final_success_rates), 1):
        print(f"Trial {i}: Success Rate = {success_rate:.2%}, Average Reward = {reward:.3f}")
    
    # Best trial
    best_trial = np.argmax(final_success_rates)
    print(f"\nBest Trial ({best_trial + 1}):")
    print(f"Final Success Rate: {final_success_rates[best_trial]:.2%}")
    print(f"Final Average Reward: {final_rewards[best_trial]:.3f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run 10 trials with 1000 episodes each
    all_rewards, all_steps, all_success = run_multiple_trials(num_trials=10, episodes=1000)
    
    # Plot results
    plot_multi_trial_metrics(all_rewards, all_steps, all_success)
    
    # Print statistics
    print_trial_statistics(all_rewards, all_steps, all_success)