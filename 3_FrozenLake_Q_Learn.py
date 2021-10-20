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


def play_episode(q, eps):
    alpha = 0.5
    gamma = 1.0
    state = env.reset()
    done = False
    r_s = []
    s_a = []

    while not done:
        eps_greedy = (
            lambda s, eps: np.random.choice([i for i, v in enumerate(q_values[state]) if v == q_values[state, :].max()])
            if random.random() > eps
            else np.random.randint(0, 3)
        )
        action = eps_greedy(state, eps)

        s_a.append((state, action))
        new_state, reward, done, _ = env.step(action)

        r_s.append(reward)

        q_values[state, action] += alpha * (reward + gamma * q_values[new_state, :].max() - q_values[state, action])
        state = new_state
    return s_a, r_s


def main():
    no_episodes = 1000
    plot_data = []

    # q-learning
    for eps in [0.01, 0.1, 0.5, 1.0]:
        rewards = []
        for _ in range(0, no_episodes):
            s_a, r_s = play_episode(q_values, eps)
            rewards.append(sum(r_s))

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
