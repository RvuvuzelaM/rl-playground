import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import ReinforceAgent

def play(n_games, lr):
  env = gym.make('LunarLander-v2')
  agent = ReinforceAgent(lr=lr, n_inputs=8, n_actions=4)

  episodes = n_games
  
  scores = []
  for i in range(episodes):
    score = 0
    done = False
    state = env.reset()

    while not done:
      action = agent.choose_action(state)
      state, reward, done, _ = env.step(action)
      agent.rewards.append(reward)
      score += reward

    scores.append(score)
    agent.train()
    print(i, score)
  
  env.close()

  plot_and_save_scores(scores, episodes, lr)

def plot_and_save_scores(scores, episodes, lr):
  plt.plot(np.arange(episodes), scores, linestyle=':')
  plt.title('n_games=' + str(episodes) + ',lr=' + str(lr))
  plt.xlabel('episode')
  plt.ylabel('reward')
  filename = '../results/reinforce/lunar_lander' + str(time.time()) + '.png'
  plt.savefig(filename)
