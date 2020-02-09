import gym
import numpy as np
from agent import ReinforceAgent
import matplotlib.pyplot as plt
 
env = gym.make('LunarLander-v2')
agent = ReinforceAgent(lr=0.001, n_inputs=8, n_actions=4)

epochs = 2500
 
scores = []
for i in range(epochs):
  score = 0
  rewards = 0
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

plt.plot(np.arange(epochs), scores)
plt.savefig('lunar_lander_results.png')
