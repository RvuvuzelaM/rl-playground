# rl-playground

### Example commands
```
REINFORCE
python3 main.py -env LunarLander-v2 -n_games 2500 -lr 0.0002

A2C
python3 main.py -env LunarLander-v2 -n_envs 32 -max_frames 200000 -lr 3e-4 -td_n 16 -n_hidden 256

PPO
python3 main.py -env LunarLander-v2 -n_envs 32 -max_frames 50000 -lr 1e-4 -td_n 16 -ppo_epochs 4 -mini_batch_size 4 -n_hidden 256 -load_best_pretrained_model True -best_avg_reward -200 -threshold_score 300 -test_env False
```