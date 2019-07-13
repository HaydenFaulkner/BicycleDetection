"""
Runs the end to end pipeline of:
Videos to frames
Detect on frames
Track on detections

"""
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import os
import random

from visualisation.image import cv_plot_bbox

def visualise(video_path, detections_dir, tracks_dir, stats_dir, vis_dir, img_snapshots_dir, vid_snapshots_dir,
              generate_sub_clips=False, display_boxes=False):

    colors = dict()
    for i in range(200):
        colors[i] = (int(256 * random.random()), int(256 * random.random()), int(256 * random.random()))

    os.makedirs(vis_dir, exist_ok=True)

    video_path, video_filename = os.path.split(video_path)
    txt_filename = video_filename[:-4]+'.txt'

    with open(os.path.join(stats_dir, txt_filename), 'r') as f:
        video_id, width, height, length = f.read().rstrip().split(',')

    with open(os.path.join(detections_dir, txt_filename), 'r') as f:
        detections = [line.rstrip().split(',') for line in f.readlines()]

    width = int(width)
    height = int(height)
    length = int(length)

    mult = True
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

        if int(d[0]) in d:
            detections_[int(d[0])].append(d_)
        else:
            detections_[int(d[0])] = [d_]
    detections = detections_

    with open(os.path.join(tracks_dir, txt_filename), 'r') as f:
        tracks = [line.rstrip().split(',') for line in f.readlines()]

    tracks_ = dict()
    for t in tracks:
        if mult:
            t_ = [int(t[1]),
                  float(t[2])*width, float(t[3])*height,
                  float(t[4])*width, float(t[5])*height]
        else:
            t_ = [int(t[1]),
                  float(t[2]), float(t[3]),
                  float(t[4]), float(t[5])]

        if int(t[0]) in tracks_:
            tracks_[int(t[0])].append(t_)
        else:
            tracks_[int(t[0])] = [t_]

    tracks = tracks_

    capture = cv2.VideoCapture(os.path.join(video_path, video_filename))

    # Get the total number of frames
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Might be a problem if video has no frames
    if total < 1:
        print("Check your opencv + ffmpeg installation, can't read videos!!!\n"
              "\nYou may need to install open cv by source not pip")
        return None

    assert total == length-1

    display_detections = True
    display_tracks = True
    display_trails = True
    display_confidences = True

    full_out_video = cv2.VideoWriter("%s_tracked.mp4" % os.path.join(vis_dir, video_filename[:-4]),
                                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

    current = 0
    while True:
        current += 1
        if current % int(total*.1) == 0:
            print("%d%% (%d/%d)" % (int(100*current/total)+1, current, total))

        flag, frame = capture.read()
        if flag == 0 and current < total-2:
            # print("frame %d error flag" % current)

            continue
        if frame is None:
            break
        v_height, v_width, _ = frame.shape
        assert v_height == height
        assert v_width == width

        out_frame = frame.copy()

        if display_tracks:
            if current in tracks:
                out_frame = cv_plot_bbox(out_path=None,
                                         img=out_frame,
                                         bboxes=[t[1:] for t in tracks[current]],
                                         scores=[1 for t in tracks[current]],  # todo should do in tracking code
                                         labels=[t[0] for t in tracks[current]],
                                         thresh=0,
                                         colors=colors,
                                         class_names=[])

        if display_detections:
            if current in detections:
                out_frame = cv_plot_bbox(out_path=None,
                                         img=out_frame,
                                         bboxes=[d[2:] for d in detections[current]],
                                         scores=[d[1] for d in detections[current]],
                                         labels=[d[0] for d in detections[current]],
                                         thresh=0,
                                         colors={0: (1, 255, 1)},
                                         class_names=['cyclist'])

        full_out_video.write(out_frame)

    if full_out_video is not None:
        full_out_video.release()

def main(_argv):
    # Get a list of videos to visualise
    if os.path.exists(FLAGS.videos_dir):
        videos = os.listdir(FLAGS.videos_dir)
        logging.info("Will process {} videos from {}".format(len(videos), FLAGS.videos_dir))
    else:
        logging.info("videos_dir does not exist: {}".format(FLAGS.videos_dir))
        return

    # generate frames
    for video in videos:
        visualise(os.path.join(FLAGS.videos_dir, video), FLAGS.detections_dir, FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir)


if __name__ == '__main__':
    flags.DEFINE_string('videos_dir', 'data/unprocessed',
                        'Directory containing the video files to process')
    flags.DEFINE_string('frames_dir', 'data/frames',
                        'Directory to hold the frames as images')
    flags.DEFINE_string('detections_dir', 'data/detections',
                        'Directory to save the detection files')
    flags.DEFINE_string('tracks_dir', 'data/tracks',
                        'Directory to save the track files')
    flags.DEFINE_string('stats_dir', 'data/stats',
                        'Directory to hold the video stats')
    flags.DEFINE_string('vis_dir', 'data/vis',
                        'Directory to hold the video visualisations')
    flags.DEFINE_string('img_snapshots_dir', 'data/snapshots/images',
                        'Directory to save image snapshots, if the flag --image_snapshots is used')
    flags.DEFINE_string('vid_snapshots_dir', 'data/snapshots/videos',
                        'Directory to save video snapshots, if the flag --video_snapshots is used')

    flags.DEFINE_float('track_detection_threshold', 0.99,
                       'The threshold on detections to them being tracked. Default is 0.99')

    flags.DEFINE_boolean('generate_sub_clips', False,
                         'Do you want to generate sub clip videos? Default is False')
    flags.DEFINE_boolean('display_boxes', False,
                         'Do you want to paint boxes on the sub clip videos and video snapshots? Default is False')
    flags.DEFINE_boolean('generate_image_snapshots', False,
                         'Do you want to save image snapshots for each track? Default is False')
    flags.DEFINE_boolean('generate_video_snapshots', False,
                         'Do you want to save video snapshots for each track? Default is False')

    try:
        app.run(main)
    except SystemExit:
        pass


