"""
run.mx.py contains wrapper code for training, will run mxnet trainer, will also allow use of tensorflow in future

"""

import argparse
import logging
import os
import warnings

from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import mxnet as mx

from config.functions import load_config
from data_processing.loading import load_datasets
from run.training.ssd import train as train_ssd
from run.training.faster_rcnn import train as train_frcnn


def parse_args():
    parser = argparse.ArgumentParser(description='Train Object Detection Network.')
    parser.add_argument('--cfg', type=str, required=True,
                        help="Path to the config file to use.")
    parser.add_argument('--backend', type=str, default="mx",
                        help="The backend to use: mxnet (mx) or tensorflow (tf). Currently only supports mxnet.")

    return parser.parse_args()


def mx_train(cfg_path):
    assert os.getcwd()[-16:] == 'BicycleDetection'

    # load the config file
    cfg = load_config(cfg_path)

    # set the random seed
    gutils.random.seed(cfg.seed)

    # training contexts ie. gpu or cpu
    ctx = [mx.gpu(int(i)) for i in cfg.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # setup network

    net_id = '_'.join((cfg.model.type, cfg.model.backbone.type, cfg.model.pretrained_on))
    net_name = '_'.join((cfg.run_id, net_id, cfg.data.name))
    model_path = os.path.join(cfg.model.root_dir, net_name)
    os.makedirs(model_path, exist_ok=True)

    # should just be able to use:
    # net = get_model(net_name, classes=cfg.classes, pretrained_base=True, transfer=cfg.model.pretrained_on)
    # but there is bug on in gluoncv ssd.py (line 809) where it's:
    # net = get_model('ssd_512_mobilenet1_0_' + str(transfer), pretrained=True, **kwargs)
    # rather than:
    # net = get_model('ssd_512_mobilenet1.0_' + str(transfer), pretrained=True, **kwargs)
    #
    # So we have to do it our own way for finetuning
    if cfg.model.pretrained_on == 'custom':
        net = get_model(net_id, pretrained_base=True, classes=cfg.classes)  # just finetuning
    else:
        net = get_model(net_id, pretrained=True)  # just finetuning
        # reset network to predict classes
        net.reset_class(cfg.classes)

    # do we want to resume?
    load_model_file = None
    start_epoch = 0
    if cfg.resume == -1:
        file_list = os.listdir(model_path)
        # if len(file_list) > 0:
        # todo sort and filter
        # todo take latest
        # todo form name
        if False:
            net.load_parameters(os.path.join(model_path, cfg.resume))
            start_epoch = 0
    elif cfg.resume:
        load_model_file = cfg.resume

    if load_model_file:
        net.load_parameters(os.path.join(model_path, load_model_file))
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()

    # load the dataset splits
    # train_dataset, val_dataset, test_dataset = load_datasets(os.path.join(os.getcwd(), cfg.data.root_dir),
    train_dataset, val_dataset, test_dataset = load_datasets(cfg.data.root_dir,
                                                             cfg.data.split_id, cfg.classes,
                                                             percent=.01)  # .01

    print(train_dataset.statistics())
    print(val_dataset.statistics())
    print(test_dataset.statistics())
    eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=cfg.classes)

    # training data
    if cfg.mixup:
        from gluoncv.data.mixup import MixupDetection
        train_dataset = MixupDetection(train_dataset)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(model_path, 'train.log'))
    logger.addHandler(fh)
    logger.info(cfg)

    # training
    print("TRAINING")
    if cfg.train.epochs - start_epoch > 0:
        if cfg.model.type == 'ssd_512':
            train_ssd(net, train_dataset, train_dataset, eval_metric, ctx, logger, start_epoch, cfg, model_path)
        elif cfg.model.type == 'faster_rcnn':
            train_frcnn(net, train_dataset, train_dataset, eval_metric, ctx, logger, start_epoch, cfg, model_path)

    # evaluate?


if __name__ == '__main__':

    args = parse_args()

    if args.backend in ['mx', 'mxnet']:
        mx_train(args.cfg)
    else:
        print("only mxnet supported at this stage.")
