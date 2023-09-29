import argparse
from tqdm import tqdm

from simpleyacs.run import run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', nargs='+', help='experiment config file(s) to run')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for exp in tqdm(args.experiment, unit='experiments'):
        run(exp)
