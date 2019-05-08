""""""
import os
import mxnet as mx
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage

from visualisation.image import pil_plot_bbox

CWD = os.getcwd()


def transform_test(imgs, short, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays. This is similar to:

    gcv.data.transforms.presets.ssd.transform_test() but the orig image isn't squashed and the x is squashed square

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        orig_img = img.asnumpy().astype('uint8')
        img = timage.imresize(img, short, short, interp=9)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def evaluate(net, dataset, ctx, eval_metric, vis=5):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for x, y in dataset:  # gets a single sample

        x, image = transform_test(x, 512)
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
            pil_plot_bbox(out_path="/media/hayden/UStorage/CODE/BicycleDetection/models/001_ssd_512_cycle/test_%03d.png" % vis,
                          img=image,
                          bboxes=bboxes[0].asnumpy(),
                          scores=scores[0].asnumpy(),
                          labels=ids[0].asnumpy(),
                          thresh=0.5,
                          class_names=['cycle'])
        # update metric
        eval_metric.update([bboxes.clip(0, x.shape[2])], [ids], [scores], gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()



def evaluateb(net, dataset, ctx, eval_metric, vis=5):
    vis_org = vis

    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for x, y in dataset:  # gets a single sample

        x, image = transform_test(x, 512)
        x = x.copyto(ctx[0])

        # get prediction results
        ids, scores, bboxes = net(x)

        # gt_ids = mx.nd.array([[[yi[4]] for yi in y]])
        gt_ids = mx.nd.array([[[1] for yi in y]])
        gt_bboxes= mx.nd.array([[[yii for yii in yi[:4]] for yi in y]])
        gt_difficults = [[yi[5]] if len(yi) > 5 else [None] for yi in y]  # put None in list to prevent cat error in voc_detection.py

        oh, ow, _ = image.shape
        bboxes[0] = tbbox.resize(bboxes[0], in_size=(512, 512), out_size=(ow, oh))
        if vis > 0:
            vis -= 1
            pil_plot_bbox(out_path="/media/hayden/UStorage/CODE/BicycleDetection/models/ssd_512_resnet50_v1_voc/test_%03d.png" % (vis_org-vis),
                          img=image,
                          bboxes=bboxes[0].asnumpy(),
                          scores=scores[0].asnumpy(),
                          labels=ids[0].asnumpy(),
                          thresh=0.2,
                          class_names=net.classes)
        # update metric
        eval_metric.update([bboxes.clip(0, x.shape[2])], [ids], [scores], gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()

if __name__ == '__main__':
    from config.functions import load_config
    from gluoncv import model_zoo
    from gluoncv import utils as gutils
    from gluoncv.utils.metrics.voc_detection import VOCMApMetric
    from data_processing.dataset import CycleDataset

    cfg = load_config('/media/hayden/UStorage/CODE/BicycleDetection/configs/001.yaml')

    # set the random seed
    gutils.random.seed(cfg.seed)

    net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

    dataset = CycleDataset(root=cfg.data.root_dir, split_id=cfg.data.split_id, split="test", shuffle=False, percent=.1)

    print(dataset.statistics())

    # training contexts ie. gpu or cpu
    ctx = [mx.gpu(int(i)) for i in cfg.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net.collect_params().reset_ctx(ctx)

    eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=net.classes)

    map_name, mean_ap = evaluateb(net, dataset, ctx, eval_metric, vis=4000)
    msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    print(msg)

#     # todo bug with coco testing, need to look into, testing works fine during training
#     args = parse_args()
#
#     # fix seed for mxnet, numpy and python builtin random generator.
#     gutils.random.seed(args.seed)
#
#     # training contexts
#     ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
#     ctx = ctx if ctx else [mx.cpu()]
#
#     net_path = args.resume.split('/')
#     args.save_prefix = args.resume[:args.resume.rfind('/')]
#     net_name = net_path[-2].split('_')
#     args.network = net_name[2]
#     args.data_shape = int(net_name[1])
#     args.pre_dataset = net_name[3]
#     net_name = '_'.join((net_name[0], net_name[1], net_name[2], net_name[3]))
#
#     # should just be able to use:
#     # net = get_model(net_name, classes=['stumps'], pretrained_base=True, transfer='voc')
#     # but there is bug on in gluoncv ssd.py (line 809) where it's:
#     # net = get_model('ssd_512_mobilenet1_0_' + str(transfer), pretrained=True, **kwargs)
#     # rather than:
#     # net = get_model('ssd_512_mobilenet1.0_' + str(transfer), pretrained=True, **kwargs)
#     #
#     # So we have to do it out own way for finetuning
#     if args.pre_dataset == 'custom':
#         net = get_model(net_name, pretrained_base=True, classes=['stumps'])  # just finetuning
#     else:
#         net = get_model(net_name, pretrained=False)  # just finetuning
#         # reset network to predict stumps
#         net.reset_class(['stumps'])
#
#     net.load_parameters(args.resume.strip())
#     # load the data
#     train_dataset, val_dataset, test_dataset, eval_metric = get_dataset(args.dataset, CWD)
#     train_data, val_data, test_data = get_dataloader(net, train_dataset, val_dataset, test_dataset,
#                                                      args.data_shape, args.batch_size, args.num_workers)
#
#     # testing and vis
#     print("TESTING: %s" % args.resume.strip())
#     net.collect_params().reset_ctx(ctx)  # put on the gpu
#     mean_ap = evalutate(net, test_dataset, test_data, eval_metric, ctx, args, logger=None)

