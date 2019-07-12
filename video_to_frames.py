"""
Runs videos to frames
"""
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import os
from tqdm import tqdm


def video_to_frames(video_path, frames_dir, overwrite=True):

    video_path, video_filename = os.path.split(video_path)

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    if os.path.exists(os.path.join(frames_dir, video_filename)) and not overwrite:
        logging.info("{} exists, won't overwrite.".format(os.path.join(frames_dir, video_filename)))
        return os.path.join(frames_dir, video_filename)
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    # load the video, ensure its valid
    capture = cv2.VideoCapture(os.path.join(video_path, video_filename))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:  # Might be a problem if video has no frames
        logging.error("Video has no frames. Check your opencv + ffmpeg installation, can't read videos!!!\n"
                      "You may need to install open cv by source not pip")

        return None

    # Let's go through the video and save the frames
    current = 0
    while True:
        flag, frame = capture.read()
        current += 1
        if flag == 0 and current < total-2:
            # print("frame %d error flag" % current)
            continue
            #break
        if frame is None:
            break
        height, width, _ = frame.shape
        cv2.imwrite(os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(current)), frame)

    return os.path.join(frames_dir, video_filename)


def main(_argv):
    # Get a list of videos to process
    if os.path.exists(FLAGS.videos_dir):
        videos = os.listdir(FLAGS.videos_dir)
        logging.info("Will process {} videos from {}".format(len(videos), FLAGS.videos_dir))
    else:
        logging.info("videos_dir does not exist: {}".format(FLAGS.videos_dir))
        return

    # generate frames
    for video in tqdm(videos, desc='Generating frames'):
        video_to_frames(os.path.join(FLAGS.videos_dir, video), FLAGS.frames_dir)


if __name__ == '__main__':
    flags.DEFINE_string('videos_dir', 'data/unprocessed',
                        'Directory containing the video files to process')
    flags.DEFINE_string('frames_dir', 'data/frames',
                        'Directory to hold the frames as images')

    try:
        app.run(main)
    except SystemExit:
        pass


