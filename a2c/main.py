import argparse
import gym
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import A2CAgent
from multiprocess_env import SubprocVecEnv

def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

def test(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        dist, _ = agent.step(state)
        state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        total_reward += reward
    return total_reward

def plot_and_save_scores(scores, frames, args):
    plt.plot(np.arange(frames), scores)
    
    plt.title('Actor Critic')
    plt.xlabel('frames')
    plt.ylabel('reward')

    DIR_PATH = '../results/a2c/lunar_lander' + str(time.time())
    os.mkdir(DIR_PATH)

    hyperparams_path = DIR_PATH + '/hyperparameters.txt'
    os.mknod(hyperparams_path)

    f = open(hyperparams_path, 'w+')
    for k, v in vars(args).items():
        f.write(str(k) + ' ' + str(v) + '\n')
    f.close()

    plot_path = DIR_PATH + '/plot.png'
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL playground')
    parser.add_argument('-env', type=str, default='LunarLander-v2')
    parser.add_argument('-n_envs', type=int, default=16)
    parser.add_argument('-max_frames', type=int, default=100000)
    parser.add_argument('-lr', type=float, default=0.0003)
    parser.add_argument('-td_n', type=int, default=16)
    parser.add_argument('-n_hidden', type=int, default=128)

    args = parser.parse_args()

    env = gym.make(args.env)
    envs = SubprocVecEnv([make_env(args.env) for i in range(args.n_envs)])

    n_inputs  = envs.observation_space.shape[0]
    n_outs = envs.action_space.n

    agent = A2CAgent(lr=args.lr, n_inputs=n_inputs, n_hidden=args.n_hidden, n_outs=n_outs, td_n=args.td_n)

    scores = []
    state = envs.reset()
    next_state = None

    idx = 0
    while idx < args.max_frames:
        agent.clear_mem()

        for _ in range(args.td_n):
            dist, value = agent.step(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            agent.entropy += dist.entropy().mean()

            agent.append_log_prob(log_prob)
            agent.append_value(value)
            agent.append_reward(reward)
            agent.append_done(done)

            state = next_state
            idx += 1

            if idx % 1000 == 0:
                score = np.mean([agent.test_env(env) for _ in range(10)])
                print(idx, score)
                scores.append(score)

        agent.train(next_state)

    plot_and_save_scores(scores, args.max_frames/1000, args)
