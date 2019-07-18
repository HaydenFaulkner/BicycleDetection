"""
Used to visualise tracks and detections

"""
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import math
import numpy as np
import os
import queue
import random
from tqdm import tqdm

from visualisation.image import cv_plot_bbox


def visualise(video_path, detections_dir, tracks_dir, stats_dir, vis_dir,
              img_snapshots_dir, vid_snapshots_dir,
              display_tracks=True, display_detections=True, display_trails=True, save_static_trails=True,
              generate_image_snapshots=True, generate_video_snapshots=True):

    video_path = os.path.normpath(video_path)
    detections_dir = os.path.normpath(detections_dir)
    tracks_dir = os.path.normpath(tracks_dir)
    stats_dir = os.path.normpath(stats_dir)
    vis_dir = os.path.normpath(vis_dir)
    img_snapshots_dir = os.path.normpath(img_snapshots_dir)
    vid_snapshots_dir = os.path.normpath(vid_snapshots_dir)

    if not generate_image_snapshots:
        img_snapshots_dir = None
    if not generate_video_snapshots:
        vid_snapshots_dir = None

    colors = dict()
    for i in range(200):
        colors[i] = (int(256 * random.random()), int(256 * random.random()), int(256 * random.random()))

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(img_snapshots_dir, exist_ok=True)
    os.makedirs(vid_snapshots_dir, exist_ok=True)

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
        logging.error("Check your opencv + ffmpeg installation, can't read videos!!!\n"
                      "\nYou may need to install open cv by source not pip")
        return None

    assert total == length-1 or total == length or total == length+1

    if display_detections or display_tracks:
        full_out_video = cv2.VideoWriter("%s_tracked.mp4" % os.path.join(vis_dir, video_filename[:-4]),
                                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

    track_trails = queue.Queue(maxsize=50)
    if save_static_trails:
        static_track_trails = {}
        avg_frame = np.zeros((1, height, width, 3), dtype=int)

    img_track_snapshots = {}
    vid_track_snapshots = {}

    for current in tqdm(range(1, length), desc="Visualising video: {}".format(video_filename)):

        flag, frame = capture.read()
        if flag == 0:
            # print("frame %d error flag" % current)
            continue
        if frame is None:
            break
        if save_static_trails and avg_frame.shape[0] < 250 and current % 50 == 0 and frame.shape == avg_frame.shape[1:]:
            avg_frame = np.vstack((avg_frame, np.expand_dims(frame, 0)))

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

                if display_trails:
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

                if save_static_trails:
                    for t in tracks[current]:
                        # make trails per track
                        if t[0] in static_track_trails:
                            static_track_trails[t[0]].append((int(t[2] + ((t[4] - t[2]) / 2)), int(t[5])))
                        else:
                            static_track_trails[t[0]] = [(int(t[2] + ((t[4] - t[2]) / 2)), int(t[5]))]

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

        if display_detections or display_tracks:
            full_out_video.write(out_frame)

        if img_snapshots_dir and current in tracks:
            for t in tracks[current]:
                # save the part of the frame containing the cyclist
                x1, y1, x2, y2 = int(t[2]), int(t[3]), int(t[4]), int(t[5])
                if t[0] not in img_track_snapshots:
                    img_track_snapshots[t[0]] = frame[y1:y2, x1:x2, :]
                else:
                    h, w, _ = img_track_snapshots[t[0]].shape
                    # replace if has a bigger area, we are assuming the bigger the better
                    if (x2 - x1) * (y2 - y1) > w * h:
                        img_track_snapshots[t[0]] = frame[y1:y2, x1:x2, :]

        if vid_snapshots_dir and current in tracks:
            # if the track wasn't found it has died, close it's associated clip
            current_tracks = [t[0] for t in tracks[current]]
            del_tids = []
            for k, vid in vid_track_snapshots.items():
                if k not in current_tracks:
                    del_tids.append(k)
                    vid.release()

            for k in del_tids:
                del vid_track_snapshots[k]

            # write out frames and add new clip if needed
            for t in tracks[current]:
                tid = t[0]
                # open a new clip for this track
                if tid not in vid_track_snapshots:
                    vid_track_snapshots[tid] = cv2.VideoWriter(
                        "%s_%d.mp4" % (os.path.join(vid_snapshots_dir, video_filename[:-4]), tid),  # open_vids),
                        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

                # get a clean frame
                clean_frame = frame.copy()

                # write out a frame
                clean_frame = cv_plot_bbox(out_path=None,
                                           img=clean_frame,
                                           bboxes=[t[2:]],
                                           scores=[t[1]],
                                           labels=[tid],
                                           thresh=0,
                                           colors=colors,
                                           class_names=[str(tid)])

                vid_track_snapshots[tid].write(clean_frame)

    if display_detections or display_tracks and full_out_video is not None:
        full_out_video.release()

    # write out the snapshot images
    if img_snapshots_dir:
        for tid, img in img_track_snapshots.items():
            out_path = os.path.join(img_snapshots_dir, "{}_{:03d}.jpg".format(video_filename[:-4], tid))
            cv2.imwrite(out_path, img)

    # release remaining snapshot track clips
    if vid_snapshots_dir:
        for k, vid in vid_track_snapshots.items():
            vid.release()

    if save_static_trails:
        avg_frame = np.mean(avg_frame, axis=0)
        for tid, dots in static_track_trails.items():
            for dot in dots:
                cv2.circle(avg_frame, dot, 2, colors[tid], -1)
        #avg_frame = cv2.addWeighted(overlay, alpha, avg_frame, 1, 0)

        cv2.imwrite("%s_trails.jpg" % os.path.join(vis_dir, video_filename[:-4]), avg_frame)


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
        visualise(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.detections_dir,
                  FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir,
                  FLAGS.display_tracks, FLAGS.display_detections, FLAGS.display_trails, FLAGS.save_static_trails,
                  FLAGS.generate_image_snapshots, FLAGS.generate_video_snapshots)


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


