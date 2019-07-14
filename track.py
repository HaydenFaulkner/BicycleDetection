"""
Runs the tracker on detections

"""
from absl import app, flags, logging
from absl.flags import FLAGS
from filterpy.kalman import KalmanFilter
import numpy as np
import os.path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


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

        self._conf = [-1]
        if len(bbox) > 4:
            self._conf = [bbox[0]]
            bbox = bbox[1:]

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

        if len(bbox) > 4:
            self._conf.append(bbox[0])
            bbox = bbox[1:]

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
        conf = np.mean(self._conf)
        box = convert_x_to_bbox(self.kf.x)[0]
        box = np.append(box, conf)
        return box


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
            d = trk.get_state()
            if trk.hits >= self.min_hits or self.det_count <= self.min_hits:
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:# or (d[2]-d[0]) < 20 or (d[3]-d[1]) < 20 or d[0] < 0 or d[1] < 0 or d[2] > img_size[0] or d[3] > img_size[1]:  # remove small box requires mult to be true so remove for now
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))


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
            iou_matrix[d, t] = iou(det[1:], trk) # ignore the confidence


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


def track(files, detections_dir, stats_dir, tracks_dir,
          every, track_detection_threshold):

    os.makedirs(tracks_dir, exist_ok=True)

    for file in tqdm(files, desc='Running tracker'):
        max_age = 30
        min_hits = 2
        tracker = Sort(max_age=max_age, min_hits=min_hits)

        with open(os.path.join(detections_dir, file), 'r') as f:
            detections = [line.rstrip().split(',') for line in f.readlines()]

        with open(os.path.join(stats_dir, file), 'r') as f:
            video_id, width, height, length = f.read().rstrip().split(',')

        width = int(width)
        height = int(height)
        length = int(length)

        mult = False
        detections_ = dict()
        for d in detections:
            if mult:
                d_ = [int(d[1]), float(d[2]),
                      float(d[3])*width, float(d[4])*height,
                      float(d[5])*width, float(d[6])*height]
            else:
                d_ = [int(d[1]), float(d[2]),
                      float(d[3]), float(d[4]),
                      float(d[5]), float(d[6])]

            if d_[1] > track_detection_threshold:
                if int(d[0]) in d:
                    detections_[int(d[0])].append(d_[1:])  # add only the bbox, dropping the det class id
                else:
                    detections_[int(d[0])] = [d_[1:]]

        KalmanBoxTracker.count = 0
        tracks = list()
        for current in range(1, length+1):

            dets = []
            det = False

            if current % every == 1:
                det = True
                if current in detections_:
                    dets = detections_[current]

            for t in tracker.update(dets, det, (width, height)):
                tracks.append([current, int(t[5]), float(t[4]), float(t[0]), float(t[1]), float(t[2]), float(t[3])])

        with open(os.path.join(tracks_dir, file), 'w') as f:
            for t in tracks:
                f.write("{},{},{},{},{},{},{}\n".format(t[0], t[1], t[2], t[3], t[4], t[5], t[6]))


def main(_argv):
    # Get a list of detections to process
    if os.path.exists(FLAGS.detections_dir):
        detections = os.listdir(FLAGS.detections_dir)
        logging.info("Will process {} detections files from {}".format(len(detections), FLAGS.detections_dir))
    else:
        logging.info("detections_dir does not exist: {}".format(FLAGS.detections_dir))
        return

    track(detections, FLAGS.detections_dir, FLAGS.stats_dir, FLAGS.tracks_dir,
          FLAGS.detect_every, FLAGS.track_detection_threshold)


if __name__ == '__main__':
    flags.DEFINE_string('stats_dir', 'data/stats',
                        'Directory to hold the video stats')
    flags.DEFINE_string('detections_dir', 'data/detections',
                        'Directory to save the detection files')
    flags.DEFINE_string('tracks_dir', 'data/tracks',
                        'Directory to save the track files')

    flags.DEFINE_integer('detect_every', 5,
                         'The frame interval to perform detection. Default is 5')
    flags.DEFINE_float('track_detection_threshold', 0.5,
                       'The threshold on detections to them being tracked. Default is 0.5')


    try:
        app.run(main)
    except SystemExit:
        pass


