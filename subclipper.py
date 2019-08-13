"""
Subclips long videos into shorter videos based on detections

"""
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import math
import os
import queue
import random
from tqdm import tqdm

from video_to_frames import video_to_frames
from detect import detect_wrapper
from track import track
from visualisation.image import cv_plot_bbox


def subclip(video_path, detections_dir, tracks_dir, stats_dir, clip_dir, around='detections',
            display_detections=False, display_tracks=False, start_buffer=100, end_buffer=50):

    video_path = os.path.normpath(video_path)
    detections_dir = os.path.normpath(detections_dir)
    tracks_dir = os.path.normpath(tracks_dir)
    stats_dir = os.path.normpath(stats_dir)
    clip_dir = os.path.normpath(clip_dir)

    assert around == 'detections' or around == 'tracks'

    colors = dict()
    for i in range(200):
        colors[i] = (int(256 * random.random()), int(256 * random.random()), int(256 * random.random()))

    os.makedirs(clip_dir, exist_ok=True)

    video_path, video_filename = os.path.split(video_path)
    txt_filename = video_filename[:-4]+'.txt'

    if not os.path.exists(os.path.join(stats_dir, txt_filename)):
        logging.info("Stats file {} does not exist so will make it first...".format(os.path.join(stats_dir,
                                                                                                 txt_filename)))

        video_to_frames(os.path.join(os.path.normpath(FLAGS.videos_dir), video_filename), FLAGS.frames_dir,
                        FLAGS.stats_dir, overwrite=False, every=FLAGS.detect_every)

    with open(os.path.join(stats_dir, txt_filename), 'r') as f:
        video_id, width, height, length = f.read().rstrip().split(',')

    if not os.path.exists(os.path.join(detections_dir, txt_filename)):
        logging.info("Detections file {} does not exist so will make it first...".format(os.path.join(detections_dir,
                                                                                                      txt_filename)))
        detect_wrapper([video_filename])

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

    if not os.path.exists(os.path.join(tracks_dir, txt_filename)):
        logging.info("Tracks file {} does not exist so will make it first...".format(os.path.join(tracks_dir,
                                                                                                  txt_filename)))

        track([txt_filename], FLAGS.detections_dir, FLAGS.stats_dir, FLAGS.tracks_dir, FLAGS.track_detection_threshold,
              FLAGS.max_age, FLAGS.min_hits)

    with open(os.path.join(tracks_dir, txt_filename), 'r') as f:
        tracks = [line.rstrip().split(',') for line in f.readlines()]

    tracks_ = dict()
    for t in tracks:
        if mult:
            t_ = [int(t[1]), float(t[2]),
                  float(t[3])*width, float(t[4])*height,
                  float(t[5])*width, float(t[6])*height]
        else:
            t_ = [int(t[1]), float(t[2]),
                  float(t[3]), float(t[4]),
                  float(t[5]), float(t[6])]

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

    assert total == length-1 or total == length or total == length+1

    full_out_video = cv2.VideoWriter("%s_shortened.mp4" % os.path.join(clip_dir, video_filename[:-4]),
                                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

    track_trails = queue.Queue(maxsize=50)
    since = 0
    out_count = 0
    for current in tqdm(range(1, length), desc="Shortening video: {}".format(video_filename)):

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
                                         bboxes=[t[2:] for t in tracks[current]],
                                         scores=[t[1] for t in tracks[current]],  # todo should do in tracking code
                                         labels=[t[0] for t in tracks[current]],
                                         thresh=0,
                                         colors=colors,
                                         class_names=[])

                track_trails_frame = {}
                for t in tracks[current]:
                    # make trails per track
                    track_trails_frame[t[0]] = (int(t[2] + ((t[4] - t[2]) / 2)), int(t[5]))

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

        forward_buffer = False
        if around == 'tracks':
            if current not in tracks:
                since += 1
            else:
                since = 0
            for check_forward in range(current, current + start_buffer):
                if check_forward in tracks:
                    forward_buffer = True
                    break

        elif around == 'detections':
            if current not in detections:
                since += 1
            else:
                since = 0
            for check_forward in range(current, current + start_buffer):
                if check_forward in detections:
                    forward_buffer = True
                    break
        else:
            ValueError()

        if forward_buffer or since < end_buffer:
            out_count += 1
            full_out_video.write(out_frame)

    if full_out_video is not None:
        full_out_video.release()

    logging.info("\n\nOriginal video length: {}\nNew video length: {} ({}% of original)".format(
        length, out_count, int(100*float(out_count)/length)))


def main(_argv):
    # Get a list of videos to visualise
    if os.path.exists(os.path.normpath(FLAGS.videos_dir)):
        videos = os.listdir(os.path.normpath(FLAGS.videos_dir))
        logging.info("Will process {} videos from {}".format(len(videos), os.path.normpath(FLAGS.videos_dir)))
    else:
        logging.info("videos_dir does not exist: {}".format(os.path.normpath(FLAGS.videos_dir)))
        return

    # generate frames
    for video in videos:
        subclip(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.detections_dir, FLAGS.tracks_dir,
                FLAGS.stats_dir, FLAGS.clips_dir,
                FLAGS.around, FLAGS.display_detections, FLAGS.display_tracks, FLAGS.start_buffer, FLAGS.end_buffer)


if __name__ == '__main__':
    flags.DEFINE_string('videos_dir', 'data/to_shorten',
                        'Directory containing the video files to process')
    flags.DEFINE_string('frames_dir', 'data/frames',
                        'Directory to hold the frames as images')
    flags.DEFINE_string('detections_dir', 'data/detections',
                        'Directory to save the detection files')
    flags.DEFINE_string('tracks_dir', 'data/tracks',
                        'Directory to save the track files')
    flags.DEFINE_string('stats_dir', 'data/stats',
                        'Directory to hold the video stats')
    flags.DEFINE_string('clips_dir', 'data/shortened',
                        'Directory to hold the shortened clips')
    flags.DEFINE_string('around', 'detections',
                        'Base the shortening off of the detections or tracks?')

    flags.DEFINE_boolean('display_tracks', False,
                         'Do you want to save a video with the tracks? Default is True')
    flags.DEFINE_boolean('display_detections', False,
                         'Do you want to save a video with the detections? Default is True')

    flags.DEFINE_integer('start_buffer', 100,
                         'The number of frames to save pre-detection or track appearance. Default is 100')
    flags.DEFINE_integer('end_buffer', 50,
                         'The number of frames to save post-detection or track appearance. Default is 50')

    # detection params if needed
    flags.DEFINE_string('gpus', '0',
                        'GPU IDs to use. Use comma for multiple eg. 0,1. Default is 0')
    flags.DEFINE_integer('num_workers', 8,
                         'The number of workers should be picked so that it’s equal to number of cores on your machine'
                         ' for max parallelization. Default is 8')

    flags.DEFINE_integer('batch_size', 2,
                         'Batch size for detection: higher faster, but more memory intensive. Default is 2')

    flags.DEFINE_string('model_path', 'models/0001/yolo3_mobilenet1_0_cycle_best.params',
                        'Path to the detection model to use')

    flags.DEFINE_integer('detect_every', 20,
                         'The frame interval to perform detection. Default is 20')
    flags.DEFINE_float('save_detection_threshold', 0.5,
                       'The threshold on detections to them being saved to the detection save file. Default is 0.5')

    # tracking params if needed
    flags.DEFINE_float('track_detection_threshold', 0.5,
                       'The threshold on detections to them being tracked. Default is 0.5')
    flags.DEFINE_integer('max_age', 40,
                         'Maximum frames between detections before a track is deleted. Bigger means tracks handle'
                         'occlusions better but also might overstay their welcome. Default is 40')
    flags.DEFINE_integer('min_hits', 2,
                         'Minimum number of detections before a track is displayed. Default is 2')

    try:
        app.run(main)
    except SystemExit:
        pass


