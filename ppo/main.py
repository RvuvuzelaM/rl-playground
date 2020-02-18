import argparse
import gym
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import PPOAgent
from multiprocess_env import SubprocVecEnv

def make_env(env_name):
  def _thunk():
    env = gym.make(env_name)
    return env
  return _thunk

def plot_and_save_scores(scores, frames, args):
  plt.plot(np.arange(frames), scores)

  plt.title('PPO')
  plt.xlabel('iters')
  plt.ylabel('avg reward for 100 episodes')

  DIR_PATH = '../results/ppo/lunar_lander' + str(time.time())
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
  parser.add_argument('-ppo_epochs', type=int, default=4)
  parser.add_argument('-mini_batch_size', type=int, default=4)
  parser.add_argument('-n_hidden', type=int, default=256)
  parser.add_argument('-load_best_pretrained_model', type=bool, default=False)
  parser.add_argument('-threshold_score', type=int, default=200)
  parser.add_argument('-best_avg_reward', type=int, default=-200)
  parser.add_argument('-test_env', type=bool, default=False)
  args = parser.parse_args()

  env = gym.make(args.env)
  envs = SubprocVecEnv([make_env(args.env) for i in range(args.n_envs)])

  n_inputs  = envs.observation_space.shape[0]
  n_outs = envs.action_space.n

  agent = PPOAgent(lr=args.lr, n_inputs=n_inputs, n_hidden=args.n_hidden, n_outs=n_outs, td_n=args.td_n, ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size)
  if args.load_best_pretrained_model:
    agent.load_model('../models/ppo/model.pt')
    print('Loaded pretrained model')

  if args.test_env:
    state = env.reset()
    done = False
    score = 0
    while not done:
      env.render()
      dist, value = agent.step(state)

      action = dist.sample()
      state, reward, done, _ = env.step(action.cpu().numpy())
      score += reward
    print(score)
  else:
    scores = []
    state = envs.reset()
    next_state = None
    early_stop = False

    best_avg_score = args.best_avg_reward

    idx = 0
    while idx < args.max_frames and not early_stop:
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

        agent.append_state(state)
        agent.append_action(action)

        state = next_state
        idx += 1

        if idx % 1000 == 0:
          score = np.mean([agent.test_env(env) for _ in range(100)])
          print(idx, score)
          scores.append(score)
          if score > best_avg_score:
            best_avg_score = score
            agent.save_model('../models/ppo/model.pt')
            print('Saved best model')
          if score > args.threshold_score:
            early_stop = True

      agent.train(next_state)

    plot_and_save_scores(scores, args.max_frames/1000, args)
