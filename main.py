import os
import json
import argparse
import ray

import train

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_dir', required=True)
parser.add_argument('-c', '--config', type=str, default='./config.json')


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config, 'r') as f:
        config = json.load(f)
    ray.init(num_gpus=2,
             num_cpus=8, local_mode=False)
    trainer = train.DDPG.remote(config, args.out_dir)
    ray.wait([trainer.train.remote()])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
