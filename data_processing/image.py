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

    get_frames = [f for f in get_frames]

    if len(get_frames) == 1 and save_path:
        if os.path.exists(os.path.join(save_path, ("%08d.jpg" % get_frames[0]))):
            frame = cv2.imread(os.path.join(save_path, ("%08d.jpg" % get_frames[0])))
            return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])

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
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


def imgs_to_vid(frames_dir, ext='.mp4', fps=25, repeat=1, delete_frames_dir=True):
    """
    Makes a video from a directory of images, will order in alphabetical
    requires opencv
    :param frames_dir: the directory containing the image files, expects it to ONLY contain image files
    :param ext: the video extension, default .mp4
    :param fps: the fps to save the video as
    :param delete_frames_dir: boolean flag to delete the frames_dir after successful video creation, default True
    :return: the video path if successful, otherwise None
    """

    img_list = os.listdir(frames_dir)
    # for i in range(len(img_list)):
    #     if len(img_list[i]) == 12:
    #         # img_list[i] = "test_0"+img_list[i][5:]
    #         os.rename(os.path.join(frames_dir, img_list[i]), os.path.join(frames_dir, "test_0"+img_list[i][5:]))
    # return
    img_list.sort()

    # # ensure ordered correctly
    # img_order_dict = {}
    # for i in range(len(img_list)):
    #     img_name = int(img_list[i][:-4])
    #     img_order_dict["%08d" % img_name] = img_list[i]

    # img_order = sorted(img_order_dict.keys())

    # load the first image to get the frame dims
    image = cv2.imread(os.path.join(frames_dir, img_list[0]))
    height = image.shape[0]
    width = image.shape[1]

    if ext == '.mp4':
        video = cv2.VideoWriter(frames_dir+ext, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), fps, (width, height))
    elif ext == '.avi':
        video = cv2.VideoWriter(frames_dir+ext, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (width, height))

    for i in range(len(img_list)):
        image = cv2.imread(os.path.join(frames_dir, img_list[i]))  # load frame
        for _ in range(repeat):
            video.write(image)

    video.release()

    if os.path.exists(frames_dir+ext):
        if delete_frames_dir:  # when successful let's check if we want to delete first
            shutil.rmtree(frames_dir)
        return frames_dir+ext  # was successful

    shutil.rmtree(frames_dir)
    return None  # wasn't successful


if __name__ == '__main__':
    pth = '/media/hayden/UStorage/CODE/BicycleDetection/models/002_faster_rcnn_resnet50_v1b_custom_cycle/test_vis'
    files = os.listdir(pth)
    for file in files:
        if len(file) == 7:
            os.rename(pth+'/'+file, pth+'/0'+file)
    imgs_to_vid("/media/hayden/UStorage/CODE/BicycleDetection/models/002_faster_rcnn_resnet50_v1b_custom_cycle/test_vis", ext='.mp4', fps=25, repeat=3, delete_frames_dir=False)