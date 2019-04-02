"""Train SSD"""
import argparse
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model

from data_loading import get_dataset, get_dataloader
from eval_utils import validate
from test_ssd import evalutate as test

CWD = os.getcwd()


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}best.params'.format(prefix, epoch, current_map))
        with open(prefix+'best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + 'train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
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
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)

    return logger

if __name__ == '__main__':

    args = parse_args()
    save_pre = args.save_prefix
    for pre in ['custom']:#['voc', 'coco']:
        args.pre_dataset = pre
        for model_name in ['mobilenet1.0', 'resnet50_v1', 'resnet101_v2']:
            args.network = model_name
            mean_aps = []
            for i in range(10):
                args.dataset = "set_%03d" % i
                # fix seed for mxnet, numpy and python builtin random generator.
                gutils.random.seed(args.seed)

                # training contexts
                ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
                ctx = ctx if ctx else [mx.cpu()]

                # setup network
                net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.pre_dataset))
                args.save_prefix = save_pre + '/' + net_name + '_' + args.dataset + '/'
                os.makedirs(args.save_prefix, exist_ok=True)

                # should just be able to use:
                # net = get_model(net_name, classes=['stumps'], pretrained_base=True, transfer='voc')
                # but there is bug on in gluoncv ssd.py (line 809) where it's:
                # net = get_model('ssd_512_mobilenet1_0_' + str(transfer), pretrained=True, **kwargs)
                # rather than:
                # net = get_model('ssd_512_mobilenet1.0_' + str(transfer), pretrained=True, **kwargs)
                #
                # So we have to do it out own way for finetuning
                if args.pre_dataset == 'custom':
                    net = get_model(net_name, pretrained_base=True, classes=['stumps'])  # just finetuning
                else:
                    net = get_model(net_name, pretrained=True)  # just finetuning
                    # reset network to predict stumps
                    net.reset_class(['stumps'])

                # do we want to resume?
                if args.resume.strip():
                    net.load_parameters(args.resume.strip())
                else:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        net.initialize()

                # load the data
                train_dataset, val_dataset, test_dataset, eval_metric = get_dataset(args.dataset, CWD)
                train_data, val_data, test_data = get_dataloader(net, train_dataset, val_dataset, test_dataset, args.data_shape, args.batch_size, args.num_workers)

                # training
                print("TRAINING")
                if args.epochs > 0:
                    logger = train(net, train_data, val_data, eval_metric, ctx, args)