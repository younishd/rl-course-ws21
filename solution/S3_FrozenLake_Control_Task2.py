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


def play_episode(q_values, epsilon, alpha=0.5):

    state = env.reset()
    done = False
    r_s = []
    while not done:
        action = choose_action(q_values, state, epsilon)

        new_state, reward, done, _ = env.step(action)
        r_s.append(reward)

        # update q-values with Q-learning
        q_values[state, action] += alpha*(reward + max(q_values[new_state]) - q_values[state, action])
        state = new_state

    return r_s


def choose_action(q_values, state, epsilon):
    if random.random() > epsilon:
        max_indices = [i for i, v in enumerate(q_values[state]) if v == max(q_values[state])]
        return random.choice(max_indices)
    else:
        return random.randint(0, 3)


def main():
    no_episodes = 1000
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:
        rewards = []
        for i in range(0, no_episodes):
            r = play_episode(q_values, epsilon=e)
            rewards.append(sum(r))

        plot_data.append(np.cumsum(rewards))

    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Mean reward")
    for i, eps in enumerate(epsilons):
        plt.plot(range(0, no_episodes), plot_data[i], label="e=" + str(eps))
    plt.legend()
    plt.show()


main()
