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


def visualise(video_path, frames_dir, detections_dir, tracks_dir, stats_dir, vis_dir,
              img_snapshots_dir, vid_snapshots_dir, around='detections',
              start_buffer=100, end_buffer=50,
              display_tracks=False, display_detections=False, display_trails=False, save_static_trails=False,
              generate_image_snapshots=False, generate_video_snapshots=False, summary=True, full=False):

    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    detections_dir = os.path.normpath(detections_dir)
    tracks_dir = os.path.normpath(tracks_dir)
    stats_dir = os.path.normpath(stats_dir)
    vis_dir = os.path.normpath(vis_dir)
    img_snapshots_dir = os.path.normpath(img_snapshots_dir)
    vid_snapshots_dir = os.path.normpath(vid_snapshots_dir)

    # if not generate_image_snapshots:
    #     img_snapshots_dir = None
    # if not generate_video_snapshots:
    #     vid_snapshots_dir = None

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

    # load detections
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

    # load tracks
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

    # load video
    capture = cv2.VideoCapture(os.path.join(video_path, video_filename))
    #
    # # Get the total number of frames
    # total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # # Might be a problem if video has no frames
    # if total < 1:
    #     logging.error("Check your opencv + ffmpeg installation, can't read videos!!!\n"
    #                   "\nYou may need to install open cv by source not pip")
    #     return None
    #
    # assert total == length-1 or total == length or total == length+1

    summary_out_video = None
    full_out_video = None
    if summary:
        summary_out_video = cv2.VideoWriter("{}_summary.mp4".format(os.path.join(vis_dir, video_filename[:-4])),
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))
    if full:
        full_out_video = cv2.VideoWriter("{}_full.mp4".format(os.path.join(vis_dir, video_filename[:-4])),
                                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

    track_trails = queue.Queue(maxsize=50)
    if save_static_trails:
        static_track_trails = {}
        avg_frame = np.zeros((1, height, width, 3), dtype=int)

    img_track_snapshots = {}
    vid_track_snapshots = {}

    since = 0
    out_count = 0
    capture.set(1, 0)
    while_safety = 0
    for current in tqdm(range(length), desc="Visualising video: {}".format(video_filename)):

        while True:
            while_safety += 1
            flag, frame = capture.read()
            # if flag != 0 and frame is not None:
            if frame is not None:
                while_safety = 0
                break
            if while_safety > 1000:
                break

        if frame is None:  # should only occur at the end of a video
            break

        if full_out_video is not None or summary_out_video is not None:
            v_height, v_width, _ = frame.shape
            frame[-50:, -250:, :] = (0, 0, 0)
            cv2.putText(frame, '{}'.format(current), (v_width-240, v_height-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

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

            if summary and not forward_buffer and since > end_buffer:
                continue  # we don't want to save out this frame

            if save_static_trails and avg_frame.shape[0] < 250 and current % 50 == 0 and frame.shape == avg_frame.shape[1:]:
                avg_frame = np.vstack((avg_frame, np.expand_dims(frame, 0)))

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

            # write out the main frame
            if summary_out_video is not None:
                summary_out_video.write(out_frame)
            if full_out_video is not None:
                full_out_video.write(out_frame)
            out_count += 1

        if generate_image_snapshots and current in tracks:
            for t in tracks[current]:
                # save the part of the frame containing the cyclist
                x1, y1, x2, y2 = int(t[2]), int(t[3]), int(t[4]), int(t[5])
                if t[0] not in img_track_snapshots:
                    img_track_snapshots[t[0]] = frame[y1:y2, x1:x2, :]
                # else:  # never replace first one
                #     h, w, _ = img_track_snapshots[t[0]].shape
                #     # replace if has a bigger area, we are assuming the bigger the better
                #     if (x2 - x1) * (y2 - y1) > w * h and x1 > 0 and y1 > 0 and x2 < width and y2 < height:
                #         img_track_snapshots[t[0]] = frame[y1:y2, x1:x2, :]

        if generate_video_snapshots and current in tracks:
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

    if full_out_video is not None:
        full_out_video.release()
    if summary_out_video is not None:
        summary_out_video.release()
    if capture is not None:
        capture.release()

    if out_count > 0:
        logging.info("\n\nOriginal video length: {}\nNew video length: {} ({}% of original)".format(
            length, out_count, int(100*float(out_count)/length)))
    else:
        logging.info("\n\nNo output video saved")

    # write out the snapshot images
    if generate_image_snapshots:
        for tid, img in img_track_snapshots.items():
            out_path = os.path.join(img_snapshots_dir, "{}_{:03d}.jpg".format(video_filename[:-4], tid))
            cv2.imwrite(out_path, img)

    # release remaining snapshot track clips
    if generate_video_snapshots:
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
        visualise(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.frames_dir, FLAGS.detections_dir,
                  FLAGS.tracks_dir, FLAGS.stats_dir, FLAGS.vis_dir,
                  FLAGS.img_snapshots_dir, FLAGS.vid_snapshots_dir, FLAGS.around,
                  FLAGS.start_buffer, FLAGS.end_buffer,
                  FLAGS.display_tracks, FLAGS.display_detections, FLAGS.display_trails, FLAGS.save_static_trails,
                  FLAGS.generate_image_snapshots, FLAGS.generate_video_snapshots, FLAGS.summary, FLAGS.full)

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
    flags.DEFINE_string('vis_dir', 'data/summaries',
                        'Directory to hold the video visualisations')
    flags.DEFINE_string('img_snapshots_dir', 'data/snapshots/images',
                        'Directory to save image snapshots, if the flag --image_snapshots is used')
    flags.DEFINE_string('vid_snapshots_dir', 'data/snapshots/videos',
                        'Directory to save video snapshots, if the flag --video_snapshots is used')

    flags.DEFINE_string('around', 'detections',
                        'Base the shortening off of the detections or tracks?')
    flags.DEFINE_integer('start_buffer', 100,
                         'The number of frames to save pre-detection or track appearance.')
    flags.DEFINE_integer('end_buffer', 50,
                         'The number of frames to save post-detection or track appearance.')

    flags.DEFINE_boolean('display_tracks', False,
                         'Do you want to save a video with the tracks?')
    flags.DEFINE_boolean('display_detections', False,
                         'Do you want to save a video with the detections?')
    flags.DEFINE_boolean('display_trails', False,
                         'Do you want display trails after the tracks?')
    flags.DEFINE_boolean('save_static_trails', False,
                         'Do you want to save an mean image with all track trails printed?')
    flags.DEFINE_boolean('generate_image_snapshots', True,
                         'Do you want to save image snapshots for each track?')
    flags.DEFINE_boolean('generate_video_snapshots', False,
                         'Do you want to save video snapshots for each track?')
    flags.DEFINE_boolean('summary', False,
                         'Do you want to save out a summary video?')
    flags.DEFINE_boolean('full', False,
                         'Do you want to save out the full video?')

    try:
        app.run(main)
    except SystemExit:
        pass


