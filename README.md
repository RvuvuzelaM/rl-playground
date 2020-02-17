# rl-playground

### Example commands
```
REINFORCE
python3 main.py -env LunarLander-v2 -n_games 2500 -lr 0.0002

A2C
python3 main.py -env LunarLander-v2 -n_envs 32 -max_frames 200000 -lr 3e-4 -td_n 16 -n_hidden 256
```