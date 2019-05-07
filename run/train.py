"""
run.mx.py contains wrapper code for training, will run mxnet trainer, will also allow use of tensorflow in future

"""

import argparse

from run.training.mx import train as mx_train


def parse_args():
    parser = argparse.ArgumentParser(description='Train Object Detection Network.')
    parser.add_argument('--cfg', type=str, required=True,
                        help="Path to the config file to use.")
    parser.add_argument('--backend', type=str, default="mx",
                        help="The backend to use: mxnet (mx) or tensorflow (tf). Currently only supports mxnet.")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.backend in ['mx', 'mxnet']:
        mx_train(args.cfg)
    else:
        print("only mxnet supported at this stage.")