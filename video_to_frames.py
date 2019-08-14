"""
Runs videos to frames
"""
from absl import app, flags, logging
from absl.flags import FLAGS
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, video_filename, frames_dir, start=0, end=0, every=0):
    """
    extract frames from a video using the fps

    :param video_path: path to the video
    :param v_id: the video id
    :param frames: the frames as a list of ints in framenumbers
    :return:
    """
    assert os.path.exists(os.path.join(video_path, video_filename))

    # Use opencv to open the video
    capture = cv2.VideoCapture(os.path.join(video_path, video_filename))
    capture.set(1, start)
    frame = start
    while frame <= end:
        # capture.set(1, frame)
        ret, image = capture.read()
        
        if ret == 0 or image is None:
            continue
        if frame % every == 0:
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))
            if not os.path.exists(save_path):
                # Save the extracted image
                cv2.imwrite(save_path, image)
        frame += 1

    capture.release()


def video_to_frames(video_path, frames_dir, stats_dir, overwrite=False, every=1):
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    stats_dir = os.path.normpath(stats_dir)

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
    _, frame = capture.read()
    height, width, _ = frame.shape
    capture.release()

    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, video_filename[:-4]+'.txt'), 'w') as f:
        f.write('{},{},{},{}'.format(video_filename, width, height, total))

    per_device_count = int(total/FLAGS.num_workers)+1
    frame_chunks = [[i, i+per_device_count] for i in range(0, total, per_device_count)]
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)
    logging.info("Extracting frames from {}".format(video_filename))
    with ProcessPoolExecutor(max_workers=FLAGS.num_workers) as executor:

        futures = [executor.submit(extract_frames, video_path, video_filename, frames_dir, f[0], f[1], every)
                   for f in frame_chunks]

    return os.path.join(frames_dir, video_filename)


def main(_argv):
    # Get a list of videos to process
    if os.path.exists(os.path.normpath(FLAGS.videos_dir)):
        videos = os.listdir(FLAGS.videos_dir)
        logging.info("Will process {} videos from {}".format(len(videos), os.path.normpath(FLAGS.videos_dir)))
    else:
        logging.info("videos_dir does not exist: {}".format(os.path.normpath(FLAGS.videos_dir)))
        return

    # generate frames
    for video in tqdm(videos, desc='Generating frames'):
        video_to_frames(os.path.join(os.path.normpath(FLAGS.videos_dir), video), FLAGS.frames_dir, FLAGS.stats_dir,
                        every=FLAGS.every)


if __name__ == '__main__':
    flags.DEFINE_string('videos_dir', 'data/unprocessed',
                        'Directory containing the video files to process')
    flags.DEFINE_string('frames_dir', 'data/frames',
                        'Directory to hold the frames as images')
    flags.DEFINE_string('stats_dir', 'data/stats',
                        'Directory to hold the video stats')
    flags.DEFINE_integer('every', 5,
                         'The frame interval to extract frames. Default is 5')
    flags.DEFINE_integer('num_workers', 6,
                         'The number of workers should be picked so that it’s equal to number of cores on your machine'
                         ' for max parallelization. Default is 6')

    try:
        app.run(main)
    except SystemExit:
        pass


