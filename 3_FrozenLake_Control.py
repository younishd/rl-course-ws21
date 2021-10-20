import random
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)
env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

no_states = env.observation_space.n
no_actions = env.action_space.n
q_values = np.zeros((no_states, no_actions))
q_counter = np.zeros((no_states, no_actions))


def play_episode(q, eps):
    state = env.reset()
    done = False
    r_s = []
    s_a = []

    while not done:
        action = (
            lambda s: q[s, :].argmax()
            if np.random.uniform() > eps
            else np.random.choice([i for i in range(0, 4) if i != q[s, :].argmax()])
        )(state)

        s_a.append((state, action))
        state, reward, done, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s


def main():
    alpha = 0.5
    gamma = 1.0
    no_episodes = 1000
    plot_data = []

    # mc control
    for eps in [0.01, 0.1, 0.5, 1.0]:
        rewards = []
        for _ in range(0, no_episodes):
            s_a, r_s = play_episode(q_values, eps)
            rewards.append(sum(r_s))

            for i, (s, a) in enumerate(s_a):
                return_i = sum(r_s[i:])
                q_counter[s, a] += 1
                q_values[s, a] += 1 / q_counter[s, a] * (return_i - q_values[s, a])

        plot_data.append((eps, np.cumsum(rewards), "MC control (eps={})".format(eps)))

    # q-learning
    for eps in [0.01, 0.1, 0.5, 1.0]:
        rewards = []
        for _ in range(0, no_episodes):
            s_a, r_s = play_episode(q_values, eps)
            rewards.append(sum(r_s))

            for i, (s, a) in enumerate(s_a):
                q_counter[s, a] += 1
                q_values[s, a] += alpha * (r_s[i] + gamma * q_values[s, :].max() - q_values[s, a])

        plot_data.append((eps, np.cumsum(rewards), "Q-learn (eps={})".format(eps)))

    # plot the rewards
    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Sum of rewards")
    for (eps, data, label) in plot_data:
        plt.plot(range(0, no_episodes), data, label=label)
    plt.legend()
    plt.show()


main()
