import argparse

from lunar_lander import play as play_lunar_lander

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL playground')
    parser.add_argument('-env', type=str, default='LunarLander-v2')
    parser.add_argument('-n_games', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.001)
    
    args = parser.parse_args()

    if args.env == 'LunarLander-v2':
        play_lunar_lander(args.n_games, args.lr)
