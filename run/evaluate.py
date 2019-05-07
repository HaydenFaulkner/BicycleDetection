"""
run.evaluate.py contains wrapper code for training, will run mxnet evaluation, will also allow use of tensorflow in future

"""

import argparse

from run.evaluation.mx import evaluate as mx_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Test Object Detection Network.')
    parser.add_argument('--cfg', type=str, required=True,
                        help="Path to the config file to use.")
    parser.add_argument('--model', type=str, default='', required=True,
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--backend', type=str, default="mx",
                        help="The backend to use: mxnet (mx) or tensorflow (tf). Currently only supports mxnet.")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.backend in ['mx', 'mxnet']:
        mx_evaluate(args.cfg)
    else:
        print("only mxnet supported at this stage.")