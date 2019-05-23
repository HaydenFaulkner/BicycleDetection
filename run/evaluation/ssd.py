""""""
import os
import mxnet as mx
from gluoncv.data.transforms import bbox as tbbox

from run.evaluation.common import transform_test
from visualisation.image import pil_plot_bbox

CWD = os.getcwd()


def evaluate(net, dataset, ctx, eval_metric, vis=50, vis_path=None):
    vis_org = vis
    if vis_path is not None:
        os.makedirs(vis_path, exist_ok=True)

    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for x, y in dataset:  # gets a single sample
        if len(y) < 1:  # todo remove such samples from set?
            continue

        x, image = transform_test(x, 512, max_size=1024)  # todo max_size=512?
        x = x.copyto(ctx[0])

        # get prediction results
        ids, scores, bboxes = net(x)

        gt_ids = mx.nd.array([[[yi[4]] for yi in y]])
        gt_bboxes= mx.nd.array([[[yii for yii in yi[:4]] for yi in y]])
        gt_difficults = [[yi[5]] if len(yi) > 5 else [None] for yi in y]  # put None in list to prevent cat error in voc_detection.py

        oh, ow, _ = image.shape
        bboxes[0] = tbbox.resize(bboxes[0], in_size=(512, 512), out_size=(ow, oh))
        if vis > 0:
            vis -= 1
            pil_plot_bbox(out_path=os.path.join(vis_path, "%05d.png" % (vis_org-vis)),
                          img=image,
                          bboxes=bboxes[0].asnumpy(),
                          scores=scores[0].asnumpy(),
                          labels=ids[0].asnumpy(),
                          thresh=0.5,
                          class_names=net.classes)
        # update metric
        eval_metric.update([bboxes.clip(0, x.shape[2])], [ids], [scores], gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


if __name__ == '__main__':
    from config.functions import load_config
    from gluoncv import model_zoo
    from gluoncv.model_zoo import get_model
    from gluoncv import utils as gutils
    from gluoncv.utils.metrics.voc_detection import VOCMApMetric
    from data_processing.dataset import CycleDataset

    cfg = load_config('/media/hayden/UStorage/CODE/BicycleDetection/configs/001.yaml')
    load_model_file = 'best.params'

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

    net.load_parameters(os.path.join(model_path, load_model_file))

    # net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)  # pretrained model

    dataset = CycleDataset(root=cfg.data.root_dir, split_id=cfg.data.split_id, split="test", shuffle=False, percent=.1)

    print(dataset.statistics())

    # training contexts ie. gpu or cpu
    ctx = [mx.gpu(int(i)) for i in cfg.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net.collect_params().reset_ctx(ctx)

    eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=net.classes)

    map_name, mean_ap = evaluate(net, dataset, ctx, eval_metric, vis=4000, vis_path=os.path.join(model_path, "test_vis"))
    msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    print(msg)

