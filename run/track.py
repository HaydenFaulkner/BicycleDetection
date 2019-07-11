"""
Used to process a directory of videos, subclipping around all cyclist appearances

"""

import argparse
import os
import queue
import time

from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms import bbox as tbbox
import mxnet as mx
import cv2

import random
import os.path
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import argparse
from filterpy.kalman import KalmanFilter

from run.evaluation.common import transform_test
from visualisation.image import cv_plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Process a directory of videos, subclipping around individual cyclist appearances, as well as generating images of all individual cyclists')
    parser.add_argument('--dir', type=str, default="data/track",
                        help="Directory path")
    parser.add_argument('--model', type=str, default="models/002_faster_rcnn_resnet50_v1b_custom_cycle/best.params",
                        help="Model path")
    parser.add_argument('--every', type=int, default=5,
                        help="Detect every this many frames. Default is 5.")
    parser.add_argument('--boxes', type=bool, default=True,
                        help="Display bounding boxes on the processed frames.")
    parser.add_argument('--gpus', type=str, default="0",
                        help="GPU ids to use, defaults to '0', if want CPU set to ''. Use comma for multiple eg. '0,1'.")
    parser.add_argument('--threshold', type=float, default=0.99,
                        help="Threshold on detection confidence. Default is 0.99")
    parser.add_argument('--show_trails', type=bool, default=True,
                        help="Display little trails behind the tracks.")
    parser.add_argument('--img_snapshots', type=bool, default=True,
                        help="Save out a single image for each track.")
    parser.add_argument('--vid_snapshots', type=bool, default=True,
                        help="Save out individual clips for each track.")
    parser.add_argument('--max_age', type=int, default=50,
                        help="Maximum age of a missing track before it is terminated. Default is 50 frames.")
    parser.add_argument('--min_hits', type=int, default=2,
                        help="Minimum number of detection / track matches before track displayed. Default is 2.")

    # parser.add_argument('--backend', type=str, default="mx",
    #                     help="The backend to use: mxnet (mx) or tensorflow (tf). Currently only supports mxnet.")

    return parser.parse_args()


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def associate_detections_to_tracks(detections, tracks, iou_threshold=0.01):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_tracks
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            iou_matrix[d, t] = iou(det, trk)


    # matched_indices = linear_assignment(-iou_matrix)
    matched_indices = np.swapaxes(np.array(linear_sum_assignment(-iou_matrix)), 0, 1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class Tracker(object):
    count = 0

    def __init__(self, bbox):

        self.time_since_update = 0
        self.id = Tracker.count
        Tracker.count += 1
        # self.history = []
        # self.hits = 0
        # self.hit_streak = 0
        self.age = 0
        self.lookback_max = 10

        # todo how many past obvs to use, as many as possible but weight?
        self.observations = [convert_bbox_to_z(bbox).squeeze()]
        self.observation_frames = [0]
        self.vel = np.zeros(4)  # current velocity: mx, my, s, r
        self.vel_frame = 1
        self.acc = np.zeros(4)  # current acceleration: mx, my, s, r
        self.current = convert_bbox_to_z(bbox).squeeze()

    def update(self, bbox):
        bbox = convert_bbox_to_z(bbox).squeeze()
        self.observations.append(bbox)
        self.observation_frames.append(self.age)

        vels_o = []
        for i in range(-1, -min(len(self.observations), self.lookback_max) - 1, -1):
            vels_i = []
            for j in range(i-1, -min(len(self.observations), self.lookback_max) - 1, -1):
                vels_i.append((self.observations[j] - self.observations[i]) /
                              (self.observation_frames[j] - self.observation_frames[i]))
            if len(vels_i) > 0:
                vels_o.append(np.mean(vels_i, keepdims=True, axis=0).squeeze())
            if len(vels_o) > 1:
                self.acc = vels_o[0] - vels_o[1]
        self.vel = np.mean(vels_o, keepdims=True, axis=0).squeeze()

    def predict(self, det):
        # if det:
        self.time_since_update += 1

        vel = self.vel.copy()
        acc = self.acc.copy()
        cur = self.observations[-1].copy()
        for i in range(self.age-self.observation_frames[-1]):  # this is how many frames we need to model over
            vel += acc
            cur += vel
        self.current = cur

        self.age += 1

        return convert_x_to_bbox(self.current).squeeze()

    def get_state(self):
        return convert_x_to_bbox(self.current).squeeze()


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(  # state transistion matrix (x * x)
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(  # measurement function (z * x)
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.  # measurement uncertainty/noise
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        # self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self, det):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if det and self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class HTrack(object):
    def __init__(self, max_age=100, min_hits=2):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.det_count = 0

    def update(self, dets, det):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(det)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if det:
            self.det_count += 1
            matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d[0]])

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = Tracker(dets[i])
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # if trk.hit_streak >= self.min_hits or self.det_count <= self.min_hits:
            ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


class Sort(object):
    def __init__(self, max_age=30, min_hits=1):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0
        self.det_count = 0

    def update(self, dets, det, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # get predicted locations from existing tracks.
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict(det)[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        if det:
            self.det_count += 1
            matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(dets, trks)

            # update matched tracks with assigned detections
            for t, trk in enumerate(self.tracks):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d[0]])

            # create and initialise new tracks for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i])
                self.tracks.append(trk)

        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.get_state()[0]
            if trk.hits >= self.min_hits or self.det_count <= self.min_hits:
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age or (d[2]-d[0]) < 20 or (d[3]-d[1]) < 20 or d[0] < 0 or d[1] < 0 or d[2] > img_size[0] or d[3] > img_size[1]:  # remove small box
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def load_net(model_path, ctx):

    # setup network
    net_id = 'faster_rcnn_resnet50_v1b_custom'
    net = get_model(net_id, pretrained_base=True, classes=['cyclist'])  # just finetuning

    net.load_parameters(model_path)

    net.collect_params().reset_ctx(ctx)

    net.hybridize(static_alloc=True)

    return net


def process_frame(image, net, ctx):
    # currently only supports batch size 1 todo
    image = np.squeeze(image)
    image = mx.nd.array(image, dtype='uint8')
    x, _ = transform_test(image, 600, max_size=1000)
    x = x.copyto(ctx[0])

    # get prediction results
    ids, scores, bboxes = net(x)
    oh, ow, _ = image.shape
    _, _, ih, iw = x.shape
    bboxes[0] = tbbox.resize(bboxes[0], in_size=(iw, ih), out_size=(ow, oh))
    return bboxes[0].asnumpy(), scores[0].asnumpy(), ids[0].asnumpy()


def track(video_dir, out_dir, video_file, net, tracker, ctx, every=25, boxes=False, threshold=0.5, show_trails=True,
          snapshot_imgs_dir=None, snapshot_clips_dir=None):

    video_path = os.path.join(video_dir, video_file)
    # Check the video exists
    if not os.path.exists(video_path):
        print("No video file found at : " + video_path)
        return None

    # Load video
    capture = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Might be a problem if video has no frames
    if total < 1:
        print("Check your opencv + ffmpeg installation, can't read videos!!!\n"
              "\nYou may need to install open cv by source not pip")
        return None

    # initialize the list of object tracks and corresponding class
    # labels
    img_track_snapshots = {}
    vid_track_snapshots = {}

    # if ext == '.mp4':
    n_tracks = 0

    current = 0
    track_trails = queue.Queue(maxsize=50)
    colors = {}
    KalmanBoxTracker.count = 0
    open_vids = 0
    while True:
        if current % int(total*.1) == 0:
            print("%d%% (%d/%d)" % (int(100*current/total)+1, current, total))

        flag, frame = capture.read()
        if flag == 0 and current < total-2:
            # print("frame %d error flag" % current)
            current += 1
            continue
            #break
        if frame is None:
            break
        height, width, _ = frame.shape

        if current < 1:
            full_out_video = cv2.VideoWriter("%s_tracked.mp4" % os.path.join(out_dir, video_file[:-4]),
                                       cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

        tbboxes = []
        tids = []
        dets = []
        det = False
        out_frame = frame.copy()
        if current % every == 0:
            det = True
            bboxes, scores, ids = process_frame(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), net=net, ctx=ctx)

            for i, box in enumerate(bboxes):
                if scores[i] > threshold:  # we found a box
                    # boxes xmin,ymin,xmax,ymax
                    dets.append(box)

            if boxes:
                out_frame = cv_plot_bbox(out_path=None,
                                         img=out_frame,
                                         bboxes=bboxes,
                                         scores=scores,
                                         labels=ids,
                                         thresh=threshold,
                                         colors={0: (1, 255, 1)},
                                         class_names=['cyclist'])

        tracks = tracker.update(dets, det, (width, height))

        for d in tracks:
            x1, y1, x2, y2, tid = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
            tbboxes.append([x1, y1, x2, y2])
            tids.append(tid)
            n_tracks = max(n_tracks, tid)

            # make different purple colour for each different track
            if tid not in colors:
                colors[tid] = (255, 0, int(255*random.random()))

            if snapshot_imgs_dir:
                # save the part of the frame containing the cyclist
                if tid not in img_track_snapshots:
                    img_track_snapshots[tid] = frame[y1:y2, x1:x2, :]
                else:
                    h, w, _ = img_track_snapshots[tid].shape
                    # replace if has a bigger area, we are assuming the bigger the better
                    if (x2-x1) * (y2-y1) > w * h:
                        img_track_snapshots[tid] = frame[y1:y2, x1:x2, :]

            if snapshot_clips_dir:
                # open a new clip for this track
                if tid not in vid_track_snapshots:
                    open_vids += 1
                    vid_track_snapshots[tid] = cv2.VideoWriter("%s_%d.mp4" % (os.path.join(snapshot_clips_dir, video_file[:-4]), tid), #open_vids),
                                                               cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

                # # if the track wasn't found it has died, close it's associated clip
                del_tids = []
                for k, vid in vid_track_snapshots.items():
                    if k not in tids:
                        del_tids.append(k)
                        vid.release()

                for k in del_tids:
                    del vid_track_snapshots[k]

                for k, vid in vid_track_snapshots.items():
                    # get a clean frame
                    clean_frame = frame.copy()

                    if show_trails:
                        pass

                    # draw and write out frames for each separate track
                    # if k in tids:
                    idx = tids.index(k)
                    if tbboxes[idx]:
                        clean_frame = cv_plot_bbox(out_path=None,
                                                   img=clean_frame,
                                                   bboxes=[tbboxes[idx]],
                                                   scores=None,
                                                   labels=[tid],
                                                   thresh=threshold,
                                                   colors=colors,
                                                   class_names=list(set([tid])))

                    vid_track_snapshots[tid].write(clean_frame)

        if show_trails:
            # make trails per track
            track_trails_frame = {}
            for d in tracks:
                track_trails_frame[int(d[4])] = (int(d[0]+((d[2]-d[0])/2)), int(d[3]))

            # put in the queue that exists over frames
            if track_trails.full():
                track_trails.get()
            track_trails.put(track_trails_frame)

            # draw the trail as dots that fade with time
            for i, trails in enumerate(list(track_trails.queue)):
                alpha = math.pow(i / len(list(track_trails.queue)), 2)
                overlay = out_frame.copy()
                for tid, dot in trails.items():
                    cv2.circle(overlay, dot, 2, colors[tid], -1)
                out_frame = cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0)

        if tbboxes:
            out_frame = cv_plot_bbox(out_path=None,
                                     img=out_frame,
                                     bboxes=tbboxes,
                                     scores=None,
                                     labels=tids,
                                     thresh=threshold,
                                     colors=colors,
                                     class_names=list(set(tids)))

        # write out main frame and go again
        full_out_video.write(out_frame)
        current += 1

        # if current > 200:
        #     break
    # release the full video
    if full_out_video is not None:
        full_out_video.release()

    # write out the snapshot images
    if snapshot_imgs_dir:
        for tid, img in img_track_snapshots.items():
            out_path = os.path.join(snapshot_imgs_dir, "{}_{:03d}.jpg".format(video_file[:-4], tid))
            cv2.imwrite(out_path, img)

    # release remaining snapshot track clips
    if snapshot_clips_dir:
        for k, vid in vid_track_snapshots.items():
            vid.release()

    if n_tracks < 1:
        print("No Cyclists tracked")
    return n_tracks, total


def tracker(video_dir, model_path, every=25, gpus='', boxes=False, threshold=0.5, show_trails=True, img_snapshots=True,
            vid_snapshots=True, max_age=30, min_hits=1):

    file_types = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']

    # Ensure we are in the BicycleDetection working directory
    if not os.getcwd()[-16:] == 'BicycleDetection':
        print("ERROR: Please ensure 'BicycleDetection' is the working directory")
        return None

    gutils.random.seed(233)

    # contexts ie. gpu or cpu
    ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net = load_net(model_path=model_path, ctx=ctx)

    out_dir = os.path.join(video_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(video_dir, 'processed'), exist_ok=True)

    snapshot_imgs_dir = None
    snapshot_vids_dir = None
    if img_snapshots:
        snapshot_imgs_dir = os.path.join(video_dir, 'img_snapshots')
        os.makedirs(snapshot_imgs_dir, exist_ok=True)
    if vid_snapshots:
        snapshot_vids_dir = os.path.join(video_dir, 'vid_snapshots')
        os.makedirs(snapshot_vids_dir, exist_ok=True)

    out_total = 0
    total_total = 0
    start_time = time.time()
    total_vids_done = 0
    vids = os.listdir(os.path.join(video_dir, 'unprocessed', gpus))
    total_vids = len(vids)
    for i, video_file in enumerate(vids):
        t = time.time()
        if video_file[-4:] not in file_types:
            print('File type %s not supported' % video_file[-4:])
            continue

        n_tracks, total = track(video_dir=os.path.join(video_dir, 'unprocessed', gpus),
                                out_dir=out_dir, video_file=video_file, net=net,
                                tracker=Sort(max_age=max_age, min_hits=min_hits), ctx=ctx, every=every,
                                boxes=boxes, threshold=threshold, show_trails=show_trails,
                                snapshot_imgs_dir=snapshot_imgs_dir, snapshot_clips_dir=snapshot_vids_dir)

        # n_tracks, total = track(video_dir=os.path.join(video_dir, 'unprocessed', gpus),
        #                         out_dir=out_dir, video_file=video_file, net=net,
        #                         tracker=HTrack(), ctx=ctx, every=every,
        #                         boxes=boxes, threshold=threshold, show_trails=show_trails,
        #                         snapshot_imgs_dir=snapshot_imgs_dir, snapshot_clips_dir=snapshot_vids_dir)

        # move video to processed dir
        if n_tracks > 0:
            total_vids_done += 1
            out_total += n_tracks
            total_total += total
            os.rename(os.path.join(video_dir, 'unprocessed', gpus, video_file), os.path.join(video_dir, 'processed', video_file))
        else:
            print("No detections found in video, consider lowering the threshold.")

        print("Processing Video %d of %d Complete (%s). Took %d minutes." % (i+1,
                                                                            total_vids,
                                                                            video_file,
                                                                            int((time.time() - t)/60.0)))

    print("Processed %d of %d Videos. Took %d minutes." % (total_vids_done,
                                                           total_vids,
                                                           int((time.time() - start_time)/60.0)))


if __name__ == '__main__':

    args = parse_args()
    tracker(video_dir=args.dir,
            model_path=args.model,
            every=args.every,
            gpus=args.gpus,
            boxes=args.boxes,
            threshold=args.threshold,
            show_trails=args.show_trails,
            img_snapshots=args.img_snapshots,
            vid_snapshots=args.vid_snapshots,
            max_age=args.max_age,
            min_hits=args.min_hits)


# todo fix bug writing out video clip snapshot with two frames of a track flip, probably something to do with the dict deleting and readding...