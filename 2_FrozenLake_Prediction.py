import random
import gym
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)
env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()


def play_episode(policy=None):
    policy = policy if policy else lambda _: random.randint(0, 3)
    state = env.reset()
    done = False
    states = [state]
    actions = []
    rewards = []
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        actions.append(action)
        states.append(state)
        rewards.append(reward)
    return states, actions, rewards


def main():
    q = np.zeros(shape=(env.env.nS, env.env.nA))
    n = np.zeros(shape=(env.env.nS, env.env.nA))

    successful_episodes = 100
    ep = 1
    while ep <= successful_episodes:
        states, actions, rewards = play_episode()

        for i, a in enumerate(actions):
            s = states[i]
            g = sum(rewards[i:])
            n[s, a] += 1
            q[s, a] += 1 / n[s, a] * (g - q[s, a])

        if sum(rewards) > 0:
            print("episode:\t{}".format(ep))
            print("q-values:")
            print(q)

            greedy_policy = lambda s: np.random.choice(np.flatnonzero(q[s, :] == q[s, :].max()))
            total_rewards = []
            for _ in range(100):
                states, actions, rewards = play_episode(policy=greedy_policy)
                total_rewards.append(sum(rewards))

            print("average greedy reward:\t{}".format(np.mean(total_rewards)))
            print()

            ep += 1


main()
