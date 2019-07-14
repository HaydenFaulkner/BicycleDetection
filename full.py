"""
Runs the end to end pipeline of:
Videos to frames
Detect on frames
Track on detections

"""
from absl import app, flags, logging
from absl.flags import FLAGS

import mxnet as mx
import os
from tqdm import tqdm

from video_to_frames import video_to_frames
from detect import prep_net, prep_data, detect
from track import track
from visualise import visualise


def main(_argv):
    if FLAGS.per_video:
        per_video()
    else:
        per_process()


def per_video():

    # Get a list of videos to process
    if os.path.exists(FLAGS.videos_dir):
        videos = os.listdir(FLAGS.videos_dir)
        logging.info("Will process {} videos from {}".format(len(videos), FLAGS.videos_dir))
    else:
        logging.info("videos_dir does not exist: {}".format(FLAGS.videos_dir))
        return

    for i, video in enumerate(videos):
        print("Video ({}) {} of {}".format(video, i, len(videos)))
        video_to_frames(os.path.join(FLAGS.videos_dir, video), FLAGS.frames_dir, FLAGS.stats_dir)

        frame_paths = list()
        for i, frame in enumerate(os.listdir(os.path.join(FLAGS.frames_dir, video))):
            if i % FLAGS.detect_every == 0:
                frame_paths.append(os.path.join(FLAGS.frames_dir, video, frame))

        ctx = [mx.gpu(int(i)) for i in FLAGS.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

        net, transform = prep_net(FLAGS.model_path, FLAGS.batch_size, ctx)

        dataset, loader = prep_data(frame_paths, transform, FLAGS.batch_size, FLAGS.num_workers)

        detect(net, dataset, loader, ctx, FLAGS.detections_dir, FLAGS.save_detection_threshold)

        track([video[:-4]+'.txt'], FLAGS.detections_dir, FLAGS.stats_dir, FLAGS.tracks_dir,
              FLAGS.detect_every, FLAGS.track_detection_threshold)

        visualise(os.path.join(FLAGS.videos_dir, video), FLAGS.detections_dir, FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir,
                  FLAGS.display_tracks, FLAGS.display_detections, FLAGS.display_trails, FLAGS.save_static_trails,
                  FLAGS.generate_image_snapshots, FLAGS.generate_video_snapshots)


def per_process():

    # Get a list of videos to process
    if os.path.exists(FLAGS.videos_dir):
        videos = os.listdir(FLAGS.videos_dir)
        logging.info("Will process {} videos from {}".format(len(videos), FLAGS.videos_dir))
    else:
        logging.info("videos_dir does not exist: {}".format(FLAGS.videos_dir))
        return

    # generate frames
    for video in tqdm(videos, desc='Generating frames'):
        video_to_frames(os.path.join(FLAGS.videos_dir, video), FLAGS.frames_dir, FLAGS.stats_dir)

    # make a frame list to build a detection dataset
    frame_paths = list()
    for video in videos:
        for i, frame in enumerate(os.listdir(os.path.join(FLAGS.frames_dir, video))):
            if i % FLAGS.detect_every == 0:
                frame_paths.append(os.path.join(FLAGS.frames_dir, video, frame))

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in FLAGS.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net, transform = prep_net(FLAGS.model_path, FLAGS.batch_size, ctx)

    dataset, loader = prep_data(frame_paths, transform, FLAGS.batch_size, FLAGS.num_workers)

    detect(net, dataset, loader, ctx, FLAGS.detections_dir, FLAGS.save_detection_threshold)

    # Get a list of detections to process
    if os.path.exists(FLAGS.detections_dir):
        detections = os.listdir(FLAGS.detections_dir)
        logging.info("Will process {} detections files from {}".format(len(detections), FLAGS.detections_dir))
    else:
        logging.info("detections_dir does not exist: {}".format(FLAGS.detections_dir))
        return

    track(detections, FLAGS.detections_dir, FLAGS.stats_dir, FLAGS.tracks_dir,
          FLAGS.detect_every, FLAGS.track_detection_threshold)

    # visualise
    for video in videos:
        visualise(os.path.join(FLAGS.videos_dir, video), FLAGS.detections_dir, FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir,
                  FLAGS.display_tracks, FLAGS.display_detections, FLAGS.display_trails, FLAGS.save_static_trails,
                  FLAGS.generate_image_snapshots, FLAGS.generate_video_snapshots)


if __name__ == '__main__':
    flags.DEFINE_boolean('per_video', True,
                         'Run everything per video, vs per process? Default is True')

    flags.DEFINE_string('videos_dir', 'data/unprocessed',
                        'Directory containing the video files to process')
    flags.DEFINE_string('frames_dir', 'data/frames',
                        'Directory to hold the frames as images')
    flags.DEFINE_string('stats_dir', 'data/stats',
                        'Directory to hold the video stats')
    flags.DEFINE_string('detections_dir', 'data/detections',
                        'Directory to save the detection files')
    flags.DEFINE_string('vis_dir', 'data/vis',
                        'Directory to hold the video visualisations')
    flags.DEFINE_string('tracks_dir', 'data/tracks',
                        'Directory to save the track files')
    flags.DEFINE_string('img_snapshots_dir', 'data/snapshots/images',
                        'Directory to save image snapshots, if the flag --image_snapshots is used')
    flags.DEFINE_string('vid_snapshots_dir', 'data/snapshots/videos',
                        'Directory to save video snapshots, if the flag --video_snapshots is used')

    flags.DEFINE_string('gpus', '0',
                        'GPU IDs to use. Use comma for multiple eg. 0,1. Default is 0')
    flags.DEFINE_integer('num_workers', 8,
                         'The number of workers should be picked so that it’s equal to number of cores on your machine'
                         ' for max parallelization. Default is 8')

    flags.DEFINE_integer('batch_size', 2,
                         'Batch size for detection: higher faster, but more memory intensive. Default is 2')
    flags.DEFINE_string('model_path', 'models/0001/yolo3_mobilenet1_0_cycle_best.params',
                        'Path to the detection model to use')

    flags.DEFINE_integer('detect_every', 5,
                         'The frame interval to perform detection. Default is 5')
    flags.DEFINE_float('save_detection_threshold', 0.5,
                       'The threshold on detections to them being saved to the detection save file. Default is 0.5')
    flags.DEFINE_float('track_detection_threshold', 0.5,
                       'The threshold on detections to them being tracked. Default is 0.5')

    flags.DEFINE_boolean('display_tracks', True,
                         'Do you want to save a video with the tracks? Default is True')
    flags.DEFINE_boolean('display_detections', True,
                         'Do you want to save a video with the detections? Default is True')
    flags.DEFINE_boolean('display_trails', True,
                         'Do you want display trails after the tracks? Default is True')
    flags.DEFINE_boolean('save_static_trails', True,
                         'Do you want to save an mean image with all track trails printed? Default is True')
    flags.DEFINE_boolean('generate_image_snapshots', True,
                         'Do you want to save image snapshots for each track? Default is True')
    flags.DEFINE_boolean('generate_video_snapshots', True,
                         'Do you want to save video snapshots for each track? Default is True')

    try:
        app.run(main)
    except SystemExit:
        pass


