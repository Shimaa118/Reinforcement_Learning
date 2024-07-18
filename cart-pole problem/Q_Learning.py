import gym
import numpy as np
import time

def discretize_state(observation, low, high, bins):
    if isinstance(observation, tuple):
        observation = observation[0]  # Extract the array part from the tuple

    discrete_state = tuple(
        np.digitize(observation[i], np.linspace(low[i], high[i], bins[i] + 1)) - 1
        for i in range(len(observation))
    )
    return discrete_state

def epsilon_greedy_action(Q, state, action_space, epsilon):
    if np.random.rand() < epsilon:
        return action_space.sample()  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action



def run_q_learning(env_name='CartPole-v1', bins=[20, 20, 20, 20], alpha=0.2, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    env = gym.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space

    low = observation_space.low
    high = observation_space.high

    Q = np.zeros(bins + [action_space.n])  # Q-table

    max_episodes = 50000
    min_episodes = 100
    target_reward = 195

    episode_rewards = []
    epsilon = epsilon_start
    start_time = time.time()
    conversion_episode = None

    for episode in range(1, max_episodes + 1):
        episode_reward = 0
        state = env.reset()
        done = False

        while not done:
            discrete_state = discretize_state(state, low, high, bins)
            action = epsilon_greedy_action(Q, discrete_state, action_space, epsilon)

            # Take action and observe next state, reward, done, info
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            else:
                next_state, reward, done, _ = step_result[0], step_result[1], step_result[2], {}

            episode_reward += reward

            # Update Q-value using Q-learning update rule
            next_discrete_state = discretize_state(next_state, low, high, bins)
            best_next_action = np.argmax(Q[next_discrete_state])
            Q[discrete_state + (action,)] += alpha * (reward + gamma * Q[next_discrete_state + (best_next_action,)] - Q[discrete_state + (action,)])

            state = next_state

        episode_rewards.append(episode_reward)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode >= min_episodes and np.mean(episode_rewards[-min_episodes:]) >= target_reward:
            conversion_episode = episode
            break

    env.close()
    end_time = time.time()
    conversion_time = conversion_episode if conversion_episode is not None else max_episodes

    return Q, np.mean(episode_rewards[-min_episodes:]), conversion_time


def evaluate_policy(Q, env_name='CartPole-v1', num_episodes=100):
    env = gym.make(env_name, render_mode="human")
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            discrete_state = discretize_state(state, env.observation_space.low, env.observation_space.high, [20] * env.observation_space.shape[0])
            action = np.argmax(Q[discrete_state])
            step_result = env.step(action)

            if isinstance(step_result, tuple) and len(step_result) > 0:
                state = step_result[0]
                reward = step_result[1]
                done = step_result[2]
            else:
                state, reward, done, _ = step_result
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    average_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")

    return average_reward

if __name__ == '__main__':
    q_learning, accuracy, conversion_time = run_q_learning()
    print("Q-Learning training completed.")
    print(f"Accuracy: {accuracy}")
    print(f"Conversion Time (in episodes): {conversion_time}")

    num_episodes_evaluation = 100
    avg_reward = evaluate_policy(q_learning, num_episodes=num_episodes_evaluation)
