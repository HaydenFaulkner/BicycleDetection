"""
Runs the end to end pipeline of:
Videos to frames
Detect on frames
Track on detections

TODO
incorporate subclipper
look into speed ups for only loading frames that exist in subclipper output in visualise.py
ensure avg_frame has same shape always!
"""
from absl import app, flags, logging
from absl.flags import FLAGS

import mxnet as mx
import os
import time
from tqdm import tqdm

from video_to_frames import video_to_frames
from detect import prep_net, prep_data, detect
from track import track
from visualise import visualise

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def main(_argv):
    if FLAGS.per_video:
        per_video()
    else:
        per_process()


def per_video():

    # Get a list of videos to process
    if os.path.exists(os.path.normpath(FLAGS.videos_dir)):
        videos = os.listdir(FLAGS.videos_dir)
        logging.info("Will process {} videos from {}".format(len(videos), os.path.normpath(FLAGS.videos_dir)))
    else:
        logging.info("videos_dir does not exist: {}".format(os.path.normpath(FLAGS.videos_dir)))
        return

    for i, video in enumerate(videos):
        print("Video ({}) {} of {}".format(video, i+1, len(videos)))
        video_to_frames(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.frames_dir, FLAGS.stats_dir,
                        overwrite=False, every=FLAGS.detect_every)

        with open(os.path.join(os.path.normpath(FLAGS.stats_dir), video[:-4]+'.txt'), 'r') as f:
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

        ctx = [mx.gpu(int(i)) for i in FLAGS.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

        net, transform = prep_net(os.path.normpath(model_path), FLAGS.batch_size, ctx)

        dataset, loader = prep_data(frame_paths, transform, FLAGS.batch_size, FLAGS.num_workers)

        detect(net, dataset, loader, ctx, FLAGS.detections_dir, FLAGS.save_detection_threshold)

        track([video[:-4]+'.txt'], FLAGS.detections_dir, FLAGS.stats_dir, FLAGS.tracks_dir,
              FLAGS.track_detection_threshold, FLAGS.max_age, FLAGS.min_hits)

        visualise(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.frames_dir, FLAGS.detections_dir,
                  FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir, FLAGS.around,
                  FLAGS.start_buffer, FLAGS.end_buffer,
                  FLAGS.display_tracks, FLAGS.display_detections, FLAGS.display_trails, FLAGS.save_static_trails,
                  FLAGS.generate_image_snapshots, FLAGS.generate_video_snapshots, FLAGS.summary)


def per_process():

    # Get a list of videos to process
    if os.path.exists(os.path.normpath(FLAGS.videos_dir)):
        videos = os.listdir(os.path.normpath(FLAGS.videos_dir))
        logging.info("Will process {} videos from {}".format(len(videos), os.path.normpath(FLAGS.videos_dir)))
    else:
        logging.info("videos_dir does not exist: {}".format(os.path.normpath(FLAGS.videos_dir)))
        return

    # generate frames
    for video in tqdm(videos, desc='Generating frames'):
        video_to_frames(os.path.join(os.path.normpath(FLAGS.videos_dir), video), os.path.normpath(FLAGS.frames_dir), os.path.normpath(FLAGS.stats_dir),
                        every=FLAGS.detect_every)

    # make a frame list to build a detection dataset
    frame_paths = list()
    for video in videos:
        with open(os.path.join(os.path.normpath(FLAGS.stats_dir), video[:-4]+'.txt'), 'r') as f:
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

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in FLAGS.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net, transform = prep_net(os.path.normpath(FLAGS.model_path), FLAGS.batch_size, ctx)

    dataset, loader = prep_data(frame_paths, transform, FLAGS.batch_size, FLAGS.num_workers)

    detect(net, dataset, loader, ctx, FLAGS.detections_dir, FLAGS.save_detection_threshold)

    # Get a list of detections to process
    if os.path.exists(os.path.normpath(FLAGS.detections_dir)):
        detections = os.listdir(FLAGS.detections_dir)
        logging.info("Will process {} detections files from {}".format(len(detections),
                                                                       os.path.normpath(FLAGS.detections_dir)))
    else:
        logging.info("detections_dir does not exist: {}".format(os.path.normpath(FLAGS.detections_dir)))
        return

    track(detections, FLAGS.detections_dir, FLAGS.stats_dir, FLAGS.tracks_dir, FLAGS.track_detection_threshold,
          FLAGS.max_age, FLAGS.min_hits)

    # visualise
    for video in videos:
        visualise(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.frames_dir, FLAGS.detections_dir,
                  FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir, FLAGS.around,
                  FLAGS.start_buffer, FLAGS.end_buffer,
                  FLAGS.display_tracks, FLAGS.display_detections, FLAGS.display_trails, FLAGS.save_static_trails,
                  FLAGS.generate_image_snapshots, FLAGS.generate_video_snapshots, FLAGS.summary)


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
    flags.DEFINE_string('vis_dir', 'data/summaries',
                        'Directory to hold the video visualisations')
    flags.DEFINE_string('tracks_dir', 'data/tracks',
                        'Directory to save the track files')
    flags.DEFINE_string('img_snapshots_dir', 'data/snapshots/images',
                        'Directory to save image snapshots, if the flag --image_snapshots is used')
    flags.DEFINE_string('vid_snapshots_dir', 'data/snapshots/videos',
                        'Directory to save video snapshots, if the flag --video_snapshots is used')

    flags.DEFINE_string('gpus', '0,2',
                        'GPU IDs to use. Use comma for multiple eg. 0,1. Default is 0')
    flags.DEFINE_integer('num_workers', 6,
                         'The number of workers should be picked so that it’s equal to number of cores on your machine'
                         ' for max parallelization. Default is 6')

    flags.DEFINE_integer('batch_size', 128,
                         'Batch size for detection: higher faster, but more memory intensive. Default is 2')
    flags.DEFINE_string('model', 'yolo',
    # flags.DEFINE_string('model', 'frcnn',
                        'Model to use, either yolo or frcnn')

    flags.DEFINE_integer('detect_every', 5,
                         'The frame interval to perform detection. Default is 5')
    flags.DEFINE_float('save_detection_threshold', 0.5,
                       'The threshold on detections to them being saved to the detection save file. Default is 0.5')

    flags.DEFINE_float('track_detection_threshold', 0.5,
                       'The threshold on detections to them being tracked. Default is 0.5')
    flags.DEFINE_integer('max_age', 40,
                         'Maximum frames between detections before a track is deleted. Bigger means tracks handle'
                         'occlusions better but also might overstay their welcome. Default is 40')
    flags.DEFINE_integer('min_hits', 2,
                         'Minimum number of detections before a track is displayed. Default is 2')

    flags.DEFINE_string('around', 'detections',
                        'Base the shortening off of the detections or tracks?')
    flags.DEFINE_integer('start_buffer', 100,
                         'The number of frames to save pre-detection or track appearance. Default is 100')
    flags.DEFINE_integer('end_buffer', 50,
                         'The number of frames to save post-detection or track appearance. Default is 50')

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
    flags.DEFINE_boolean('summary', True,
                         'Do you want to only save out the summary video? Default is True')

    try:
        app.run(main)
    except SystemExit:
        pass


