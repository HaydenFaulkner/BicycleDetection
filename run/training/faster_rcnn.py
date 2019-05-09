"""Train Faster-RCNN end to end."""

import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from tensorboardX import SummaryWriter


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        # pred_label = mx.nd.argmax(rpn_cls_logits, axis=1, keepdims=True)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


def get_dataloader(net, dataset, split, data_shape, batch_size, num_workers):
    """Get dataloader."""
    if split == 'train':
        train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
        return mx.gluon.data.DataLoader(
            dataset.transform(FasterRCNNDefaultTrainTransform(net.short, net.max_size, net)), batch_size,
            True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    elif split == 'val' or split == 'test':
        val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
        return mx.gluon.data.DataLoader(
            dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)), batch_size,
            False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    else:
        return None


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
                    epoch, current_map, best_map, '{:s}/best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}/best.params'.format(prefix))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}/{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}/{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()

# def evaluate(net, val_data, ctx, eval_metric):
#     """Test on validation dataset."""
#     clipper = gcv.nn.bbox.BBoxClipToImage()
#     eval_metric.reset()
#     net.hybridize(static_alloc=True)
#     for batch in val_data:
#         batch = split_and_load(batch, ctx_list=ctx)
#         det_bboxes = []
#         det_ids = []
#         det_scores = []
#         gt_bboxes = []
#         gt_ids = []
#         gt_difficults = []
#         for x, y, im_scale in zip(*batch):
#             # get prediction results
#             ids, scores, bboxes = net(x)
#             det_ids.append(ids)
#             det_scores.append(scores)
#             # clip to image size
#             det_bboxes.append(clipper(bboxes, x))
#             # rescale to original resolution
#             im_scale = im_scale.reshape((-1)).asscalar()
#             det_bboxes[-1] *= im_scale
#             # split ground truths
#             gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
#             gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
#             gt_bboxes[-1] *= im_scale
#             gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
#
#         # update metric
#         for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
#             eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
#     return eval_metric.get()

def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha

def train(net, train_dataset, val_dataset, eval_metric, ctx, logger, start_epoch, cfg, save_path):
    """Training pipeline"""
    tb_sw = SummaryWriter(
        log_dir=os.path.join(save_path, 'tb'),
        comment=cfg.run_id)

    train_dataloader = get_dataloader(net, train_dataset,
                                      split='train', data_shape=cfg.data.shape,
                                      batch_size=cfg.train.batch_size, num_workers=cfg.num_workers)

    val_dataloader = get_dataloader(net, val_dataset,
                                    split='val', data_shape=cfg.data.shape,
                                    batch_size=cfg.train.batch_size, num_workers=cfg.num_workers)

    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    trainer = gluon.Trainer(
        net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': cfg.train.learning_rate, 'wd': cfg.train.weight_decay, 'momentum': cfg.train.momentum,
         'clip_gradient': 5})

    # lr decay policy
    lr_decay = float(cfg.train.lr_decay)
    lr_steps = sorted([float(ls) for ls in cfg.train.lr_decay_epochs])
    lr_warmup = float(cfg.train.lr_warmup)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1/9.)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'),]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    # set up logger
    # if args.verbose:
    #     logger.info('Trainable parameters:')
    #     logger.info(net.collect_train_params().keys())

    logger.info('Start training from [Epoch {}]'.format(start_epoch))
    best_map = [0]
    for epoch in range(start_epoch, cfg.train.epochs):
        mix_ratio = 1.0
        # if cfg.mixup:
        #     # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
        #     train_data._dataset.set_mixup(np.random.uniform, 0.5, 0.5)
        #     mix_ratio = 0.5
        #     if epoch >= args.epochs - args.no_mixup_epochs:
        #         train_data._dataset.set_mixup(None)
        #         mix_ratio = 1.0
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_dataloader):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % cfg.log_interval == 0:
                        logger.info('[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            with autograd.record():
                for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    cls_pred, box_pred, roi, samples, matches, rpn_score, rpn_box, anchors = net(data, gt_box)
                    # losses of rpn
                    rpn_score = rpn_score.squeeze(axis=-1)
                    num_rpn_pos = (rpn_cls_targets >= 0).sum()
                    rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets, rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
                    rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos
                    # rpn overall loss, use sum rather than average
                    rpn_loss = rpn_loss1 + rpn_loss2
                    # generate targets for rcnn
                    cls_targets, box_targets, box_masks = net.target_generator(roi, samples, matches, gt_label, gt_box)
                    # losses of rcnn
                    num_rcnn_pos = (cls_targets >= 0).sum()
                    rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets, cls_targets >= 0) * cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
                    rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / box_pred.shape[0] / num_rcnn_pos
                    rcnn_loss = rcnn_loss1 + rcnn_loss2
                    # overall losses
                    losses.append(rpn_loss.sum() * mix_ratio + rcnn_loss.sum() * mix_ratio)
                    metric_losses[0].append(rpn_loss1.sum() * mix_ratio)
                    metric_losses[1].append(rpn_loss2.sum() * mix_ratio)
                    metric_losses[2].append(rcnn_loss1.sum() * mix_ratio)
                    metric_losses[3].append(rcnn_loss2.sum() * mix_ratio)
                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets>=0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[cls_targets], [cls_pred]])
                    add_losses[3].append([[box_targets, box_masks], [box_pred]])

                global_step = epoch * len(train_dataloader) + i
                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
            trainer.step(batch_size)
            # update metrics
            if cfg.log_interval and not (i + 1) % cfg.log_interval:
                for metric in metrics + metrics2:
                    tb_sw.add_scalar(tag=metric.get()[0], scalar_value=metric.get()[1], global_step=global_step)

                # msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, cfg.log_interval * batch_size/(time.time()-btic), msg))
                btic = time.time()

        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time()-tic), msg))
        if (epoch % cfg.train.val_every == 0) or (cfg.train.checkpoint_every and epoch % cfg.train.checkpoint_every == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_dataloader, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            tb_sw.add_scalar(tag='mAP', scalar_value=current_map, global_step=global_step)
        else:
            current_map = 0.
        # save_params(net, logger, best_map, current_map, epoch, cfg.checkpoint_every, save_path)

    return logger
