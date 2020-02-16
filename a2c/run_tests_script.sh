#!/bin/sh
python3 main.py -env LunarLander-v2 -n_envs 64 -max_frames 200000 -lr 3e-4 -td_n 16 -n_hidden 128
python3 main.py -env LunarLander-v2 -n_envs 64 -max_frames 200000 -lr 1e-3 -td_n 32 -n_hidden 64
python3 main.py -env LunarLander-v2 -n_envs 64 -max_frames 200000 -lr 1e-3 -td_n 32 -n_hidden 128
python3 main.py -env LunarLander-v2 -n_envs 64 -max_frames 200000 -lr 5e-4 -td_n 32 -n_hidden 128
