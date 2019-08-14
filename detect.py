"""
Runs the end to end pipeline of:
Videos to frames
Detect on frames
Track on detections

"""
from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import os
import time
from tqdm import tqdm

from video_to_frames import video_to_frames

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.dataset import Dataset
from gluoncv.data.transforms import image as timage
from gluoncv.data.batchify import Tuple, Stack
from gluoncv.model_zoo import get_model

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class DetectSet(Dataset):
    def __init__(self, file_list):
        super(DetectSet, self).__init__()
        self._file_list = file_list

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx):
        img_path = self._file_list[idx]
        img = mx.image.imread(img_path, 1)
        return img, idx

    def sample_path(self, idx):
        return self._file_list[idx]


class FasterRCNNDefaultInferenceTransform(object):
    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, sidx):
        """Apply transform to inference image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, sidx


class YOLO3DefaultInferenceTransform(object):
    """Default YOLO inference transform.
    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, sidx):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, sidx


def as_numpy(a):
    """Convert a (list of) mx.NDArray into numpy.ndarray"""
    if isinstance(a, (list, tuple)):
        out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
        try:
            out = np.concatenate(out, axis=0)
        except ValueError:
            out = np.array(out)
        return out
    elif isinstance(a, mx.nd.NDArray):
        a = a.asnumpy()
    return a


def prep_data(frame_paths, transform, batch_size, num_workers):
    dataset = DetectSet(frame_paths)
    loader = gluon.data.DataLoader(dataset.transform(transform),
                                   batch_size, False, last_batch='keep',
                                   num_workers=num_workers, batchify_fn=Tuple(Stack(),Stack()),)
    return dataset, loader


def prep_net(model_path, batch_size, ctx):

    if 'faster_rcnn' in model_path:
        assert batch_size == 1, 'can only have a batch size of 1 for faster rcnn'
        net_id = 'faster_rcnn_resnet50_v1b_custom'

        net = get_model(net_id, root='models', pretrained_base=True, classes=['cyclist'])

        transform = FasterRCNNDefaultInferenceTransform()
    elif 'yolo' in model_path:
        net_id = 'yolo3_mobilenet1.0_custom'

        net = get_model(net_id, root='models', pretrained_base=True, classes=['cyclist'])

        transform = YOLO3DefaultInferenceTransform(416, 416)

    net.load_parameters(model_path)

    net.collect_params().reset_ctx(ctx)

    return net, transform


def detect(net, dataset, loader, ctx, detections_dir, save_detection_threshold):

    detections_dir = os.path.normpath(detections_dir)  # good for win compat
    os.makedirs(detections_dir, exist_ok=True)

    net.set_nms(nms_thresh=0.45, nms_topk=400)
    # net.hybridize()
    net.hybridize(static_alloc=True)
    boxes = dict()
    with tqdm(total=len(dataset), desc='Detecting') as pbar:
        for ib, batch in enumerate(loader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            sidxs = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            sidxs_ = []
            for x, sidx in zip(data, sidxs):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                sidxs_.append(sidx)

            pbar.update(batch[0].shape[0])

            for id, score, box, sidx in zip(*[as_numpy(x) for x in [det_ids, det_scores, det_bboxes, sidxs_]]):

                valid_pred = np.where(id.flat >= 0)[0]  # get the boxes that have a class assigned
                box = box[valid_pred, :] / batch[0].shape[2]  # normalise boxes
                id = id.flat[valid_pred].astype(int)
                score = score.flat[valid_pred]
                for id_, box_, score_ in zip(id, box, score):
                    if score_ > save_detection_threshold:
                        vid_id = os.path.normpath(dataset.sample_path(int(sidx))).split(os.path.sep)[-2]
                        frame = int(os.path.normpath(dataset.sample_path(int(sidx))).split(os.path.sep)[-1][:-4])
                        if vid_id in boxes:
                            boxes[vid_id].append([frame, id_, score_]+list(box_))
                        else:
                            boxes[vid_id] = [[frame, id_, score_]+list(box_)]

    for vid_id, boxs in tqdm(boxes.items(), desc='Writing out detection files'):
        boxs.sort(key=lambda x: x[0])
        with open(os.path.join(detections_dir, vid_id[:-4] + '.txt'), 'w') as f:

            for box in boxs:
                f.write("{},{},{},{},{},{},{}\n".format(box[0], box[1], box[2], box[3], box[4], box[5], box[6]))

    return detections_dir


def detect_wrapper(videos=None):
    # Get a list of videos to process
    if os.path.exists(os.path.normpath(FLAGS.videos_dir)):
        if not videos:
            videos = os.listdir(os.path.normpath(FLAGS.videos_dir))
        logging.info("Will process {} videos from {}".format(len(videos), os.path.normpath(FLAGS.videos_dir)))
    else:
        logging.info("videos_dir does not exist: {}".format(os.path.normpath(FLAGS.videos_dir)))
        return

    # generate frames if need be, if they exist don't do
    for video in tqdm(videos, desc='Generating frames'):
        video_to_frames(os.path.join(os.path.normpath(FLAGS.videos_dir), video), os.path.normpath(FLAGS.frames_dir), os.path.normpath(FLAGS.stats_dir),
                        overwrite=False, every=FLAGS.detect_every)

    frame_paths = list()
    for video in videos:
        with open(os.path.join(FLAGS.stats_dir, video[:-4]+'.txt'), 'r') as f:
            video_id, width, height, length = f.read().rstrip().split(',')

        frame_paths = list()
        for frame in range(0, int(length), FLAGS.detect_every):
            frame_path = os.path.join(os.path.normpath(FLAGS.frames_dir), video, "{:010d}.jpg".format(frame))
            if not os.path.exists(frame_path):
                logging.warning("{} Frame image file doesn't exist. Probably because you extracted frames at "
                                "a higher 'every' value than the 'detect_every' value specified".format(frame_path))
                logging.warning("Will re-extract frames, you have 10 seconds to cancel")
                time.sleep(10)

                video_to_frames(os.path.join(os.path.normpath(FLAGS.videos_dir), video), os.path.normpath(FLAGS.frames_dir),
                                os.path.normpath(FLAGS.stats_dir), overwrite=True, every=FLAGS.detect_every)
            else:
                frame_paths.append(frame_path)

    if 'yolo' in FLAGS.model:
        model_path = 'models/0001/yolo3_mobilenet1_0_cycle_best.params'
    else:
        model_path = 'models/0002/faster_rcnn_best.params'
        FLAGS.batch_size = 1
        FLAGS.gpus = '0'

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in FLAGS.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    
    net, transform = prep_net(os.path.normpath(model_path), FLAGS.batch_size, ctx)

    dataset, loader = prep_data(frame_paths, transform, FLAGS.batch_size, FLAGS.num_workers)

    detect(net, dataset, loader, ctx, os.path.normpath(FLAGS.detections_dir), FLAGS.save_detection_threshold)


def main(_argv):
    detect_wrapper()


if __name__ == '__main__':
    flags.DEFINE_string('videos_dir', 'data/unprocessed',
                        'Directory containing the video files to process')
    flags.DEFINE_string('frames_dir', 'data/frames',
                        'Directory to hold the frames as images')
    flags.DEFINE_string('detections_dir', 'data/detections',
                        'Directory to save the detection files')
    flags.DEFINE_string('stats_dir', 'data/stats',
                        'Directory to hold the video stats')

    flags.DEFINE_string('gpus', '0,2',
                        'GPU IDs to use. Use comma for multiple eg. 0,1. Default is 0')
    flags.DEFINE_integer('num_workers', 8,
                         'The number of workers should be picked so that its equal to number of cores on your machine'
                         ' for max parallelization. Default is 8')

    flags.DEFINE_integer('batch_size', 128,
                         'Batch size for detection: higher faster, but more memory intensive. Default is 2')

    flags.DEFINE_string('model', 'yolo',
    # flags.DEFINE_string('model', 'frcnn',
                        'Model to use, either yolo or frcnn')

    flags.DEFINE_integer('detect_every', 5,
                         'The frame interval to perform detection. Default is 5')
    flags.DEFINE_float('save_detection_threshold', 0.5,
                       'The threshold on detections to them being saved to the detection save file. Default is 0.5')

    try:
        app.run(main)
    except SystemExit:
        pass


