"""
Training script for mxnet ssd object detection pipeline
"""

import os
import sys
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
import gluoncv as gcv
from tensorboardX import SummaryWriter

from run.evaluation.ssd import evaluate
from run.training.common import save_params

sys.setrecursionlimit(10000)   # set recursion limit to 10000


def get_dataloader(net, dataset, split, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    if split == 'train':
        with autograd.train_mode():
            _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))

        batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
        return gluon.data.DataLoader(dataset.transform(SSDDefaultTrainTransform(width, height, anchors)), batch_size,
                                     True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    elif split == 'val' or split == 'test':
        batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        return gluon.data.DataLoader(dataset.transform(SSDDefaultValTransform(width, height)), batch_size,
                                     False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)
    else:
        return None


def train(net, train_dataset, val_dataset, eval_metric, ctx, logger, start_epoch, cfg, save_path):
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
                tb_sw.add_scalar(tag=name1, scalar_value=loss1, global_step=global_step)
                tb_sw.add_scalar(tag=name2, scalar_value=loss2, global_step=global_step)
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % cfg.train.val_every == 0) or (cfg.train.checkpoint_every and epoch % cfg.train.checkpoint_every == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = evaluate(net, val_dataset, ctx, eval_metric, vis=100, vis_path=save_path)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            tb_sw.add_scalar(tag='mAP', scalar_value=current_map, global_step=global_step)
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, cfg.checkpoint_every, save_path)

    return logger
