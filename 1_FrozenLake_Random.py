import gym
import random

random.seed(0)
env.seed(0)

print("## Frozen Lake ##")

action2string = {0: "left", 1: "down", 2: "right", 3: "up"}


def run_episode(env, policy=False):
    policy = policy if policy else lambda s: random.randint(0, env.env.nA - 1)
    state = env.reset()
    done = False
    episode = []
    while not done:
        action = policy(state)
        new_state, reward, done, _ = env.step(action)
        episode.append({"action": action, "state": state, "new_state": new_state})
        state = new_state
        print(
            "action: {}, state: {}, reward: {}".format(
                action2string[action], state, reward
            )
        )
        env.render()
    return state, reward, done, episode


def run(env, policy=False, max_ep=False):
    global action2string

    env.render()

    ep_count = 0
    while not max_ep or ep_count < max_ep:
        ep_count += 1
        print("episode: {}".format(ep_count))
        state, reward, done, episode = run_episode(env, policy)
        if state == 15:
            break
        print("---")

    print()
    print("done after {} episodes and {} actions!".format(ep_count, len(episode)))
    for e in episode:
        print(
            "action: {}, state: {}".format(action2string[e["action"]], e["new_state"])
        )

    return episode


def policy_from_episode(env, episode):
    state_actions = {}
    for e in reversed(episode):
        if e["state"] not in state_actions:
            state_actions[e["state"]] = e["action"]
        if len(state_actions) == env.env.nS:
            break

    print(state_actions)

    return (
        lambda s: state_actions[s]
        if s in state_actions
        else random.randint(0, env.env.nA - 1)
    )


env = gym.make("FrozenLake-v1", is_slippery=False)
e = run(env)

policy = policy_from_episode(env, e)
run(env, policy=policy)

env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
run(env)

env = gym.make("FrozenLake-v1", is_slippery=True)
run(env, policy=policy)
