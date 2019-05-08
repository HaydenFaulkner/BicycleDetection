"""
Train script for mxnet object detection pipeline
"""

import os
import logging
import warnings
import sys
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from tensorboardX import SummaryWriter

from config.functions import load_config
from data_processing.loading import load_datasets, get_dataloader
from run.evaluation.mx import evaluate

CWD = os.getcwd()

sys.setrecursionlimit(10000)   # set recursion limit to 10000


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}best.params'.format(prefix, epoch, current_map))
        with open(prefix+'best_map.log', 'a') as f:
            f.write('e{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_e{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def train_loop(net, train_dataset, val_dataset, eval_metric, ctx, logger, start_epoch, cfg, save_path):
    """Training pipeline"""

    tb_sw = SummaryWriter(
        log_dir=os.path.join(save_path, 'tb'),
        comment=cfg.run_id)

    # dataloader
    train_dataloader = get_dataloader(net, train_dataset,
                                      split='train', data_shape=cfg.data.shape,
                                      batch_size=cfg.train.batch_size, num_workers=cfg.num_workers)

    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': cfg.train.learning_rate, 'wd': cfg.train.weight_decay, 'momentum': cfg.train.momentum})

    # lr decay policy
    lr_decay = float(cfg.train.lr_decay)
    lr_steps = sorted([float(ls) for ls in cfg.train.lr_decay_epochs])

    # setup losses
    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # start the training loop
    logger.info('Start training from [Epoch {}]'.format(start_epoch))
    best_map = [0]
    for epoch in range(start_epoch, cfg.train.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_dataloader):
            batch_numpy = [b.asnumpy() for b in batch]

            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            # data_numpy = data.cpu().asnumpy()
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(cls_preds, box_preds, cls_targets, box_targets)

                global_step = epoch * len(train_dataloader) + i
                tb_sw.add_scalar(tag='Training_loss', scalar_value=sum_loss[0], global_step=global_step)
                tb_sw.add_scalar(tag='Training_cls_loss', scalar_value=cls_loss[0], global_step=global_step)
                tb_sw.add_scalar(tag='Training_box_loss', scalar_value=box_loss[0], global_step=global_step)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if cfg.log_interval and not (i + 1) % cfg.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()
            break

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % cfg.train.val_every == 0) or (cfg.train.checkpoint_every and epoch % cfg.train.checkpoint_every == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = evaluate(net, val_dataset, ctx, eval_metric, vis=10)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            tb_sw.add_scalar(tag='mAP', scalar_value=current_map, global_step=global_step)
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, cfg.checkpoint_every, save_path)

    return logger


def train(cfg_path):

    # load the config file
    cfg = load_config(cfg_path)

    # set the random seed
    gutils.random.seed(cfg.seed)

    # training contexts ie. gpu or cpu
    ctx = [mx.gpu(int(i)) for i in cfg.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # setup network
    net_name = '_'.join((cfg.run_id, cfg.model.type, cfg.data.name))
    net_id = '_'.join((cfg.model.type, cfg.model.backbone.type, cfg.model.pretrained_on))
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
        # reset network to predict stumps
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
    train_dataset, val_dataset, test_dataset = load_datasets(cfg.data.root_dir, cfg.data.split_id, cfg.classes, percent=.15)

    print(train_dataset.statistics())
    print(val_dataset.statistics())
    print(test_dataset.statistics())
    eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=cfg.classes)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(model_path, 'train.log'))
    logger.addHandler(fh)
    logger.info(cfg)

    # training
    print("TRAINING")
    if cfg.train.epochs-start_epoch > 0:
        train_loop(net, train_dataset, val_dataset, eval_metric, ctx, logger, start_epoch, cfg, model_path)

    #evaluate?