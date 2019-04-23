"""
data_processing.image contains functions for processing video and image data

"""
import os
import shutil
import numpy as np
import cv2


def get_fps(video_path):
    """
    Get the fps of a video clip

    :param video_path: str, the path to the video

    :return: fps: float, the fps of the video
    """

    # Check the video exists
    if not os.path.exists(video_path):
        print("No video file found at : " + video_path)
        return None

    # Load video
    capture = cv2.VideoCapture(video_path)

    # Get the fps
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    return fps


def get_frame_list_from_timestamps(timestamps, fps):
    """
    Generate a frame_list give a list of starting and ending timestamps

    :param timestamps: list([start, end]), list of len(2) lists with start and end times
    :param fps: float, the fps of the video

    :return: frames_list: list(int), list of unique frame numbers within the timestamps
    """
    frames_list = []

    for timestamp in timestamps:
        start = timestamp[0]
        end = timestamp[1]

        frames_list += list(range(int(round(fps * start)), int(round(fps * end))))

    # make it contain unique elements
    frames_list = list(set(frames_list))

    return frames_list


def extract_frames(video_path, get_frames=-1, save_path=None):
    """
    extract frames from a video, requires opencv

    :param video_path: str, the path to the video
    :param get_frames: list(int), the frames of the video we want to get, default -1 which is all
    :param save_path: str, the path to save the frames as images, default is None which doesn't save images

    :return: frames: np.array, the images of the video (f, h, w, c) [0, 255]
    """

    get_frames = [f-12 for f in get_frames]

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

    # if we want to save the frames to a directory, make it
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # if we want all frames, then just make a list from 0 to total
    if get_frames == -1:
        get_frames = list(range(total))

    # step through the video and get the frames
    current = 0
    frames = []
    got_frames = []
    while True:
        flag, frame = capture.read()
        if flag == 0:
            break
        if current > max(get_frames):
            break

        if current in get_frames:
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            got_frames.append(current)

            # save frame to file
            if save_path:
                cv2.imwrite(os.path.join(save_path, ("%08d.jpg" % current)), frame)

        current += 1

    # ensure we got all the frames we wanted, if not...
    if len(set(get_frames).difference(set(got_frames))) != 0:
        print("Was unable to do frames : ")
        print(list(set(get_frames).difference(set(got_frames))))

        # delete the save_path dir if we made it
        if save_path and os.path.exists(save_path):
            shutil.rmtree(save_path)
        # return None

        # frames.append(np.zeros((cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_CHANNEL)))
        frames.append(np.zeros((1080, 1920, 3)))
    return np.array(frames)
