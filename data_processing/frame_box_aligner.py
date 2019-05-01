"""
Used to align videos to annotations by letting a user set a frame offset per video

Go through videos one by one asking user for a frame offset

Generates an annotation file

"""

import os
import shutil
import numpy as np
import cv2

from data_processing.annotation import load_annotation_data, interpolate_annotation

def play_video(save_path, video_path, annotations):
    """

    :param video_path: str, the path to the video
    """



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

    # Get a list of frames that have an annotation
    frames = set()
    for inst in annotations['instances'].values():
        frames |= set(inst['key_boxes'].keys())  # add set
    frames = list(sorted(frames))

    # Get the indecies of the separate instances, only really useful when boxes are already interpolated
    instance_gap_thresh = 5  # frames to split instances
    instances = []
    for i, frame_num in enumerate(frames):
        if i == 0:
            instances.append(0)
        else:
            if frame_num - past_frame_num > instance_gap_thresh:
                instances.append(i)

        past_frame_num = frame_num

    save = True
    current = 0
    offset = -13
    buffer = -1  # incase we make a mistake, only allows one, don't be reckless
    swap = {}
    print("Use a/d or the left/right arrow keys to change frame, use w/s or up/down arrow keys to change offset, e/enter to set offset, backspace to go back")
    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames[current] + offset))
        flag, frame = capture.read()
        for inst in annotations['instances'].values():
            if frames[current] in inst['key_boxes']:
                box = inst['key_boxes'][frames[current]]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # tx,ty,bx,by

        # Display the resulting frame
        cv2.imshow('t', frame)

        # waitKey
        key = cv2.waitKey(0)
        if key == 83 or key == 100:
            flag, frame = capture.read()
            if flag == 0:
                current = 0
            else:
                current += 1
        elif key == 81 or key == 97:
            current -= 1
        elif key == 84 or key == 115:
            offset -= 1
        elif key == 82 or key == 119:
            offset += 1
        elif key == 13 or key == 101:
            swap[frames[current]] = frames[current] + offset
            buffer = frames[current]
            del frames[current]
        elif key == 8:
            if buffer >= 0:
                frames.append(buffer)
                buffer = -1
                current = len(frames)-1
        elif key == 120:
            save = False
            break

        if current >= len(frames):
            current = 0

        if len(frames) == 0:
            break

    if save:
        # save out the annotation file with the offsets
        boxes = []
        for inst in annotations['instances'].values():
            for frame_num, box in inst['key_boxes'].items():
                boxes.append([swap[frame_num]] + box + [inst['name']])
        boxes.sort(key=lambda x: x[0])
        with open(save_path, 'w') as f:
            for box in boxes:
                f.write("%d,%d,%d,%d,%d,%d,%s\n" % (box[0], box[1], box[2], box[3], box[4], box[5], box[6]))


def aligner(root):
    os.makedirs(os.path.join(root, 'annotations_txt'), exist_ok=True)
    file_list = os.listdir(os.path.join(root, 'annotations'))
    for i, file in enumerate(file_list):
        print("%s (%d / %d)" % (file, i, len(file_list)))

        annotations = load_annotation_data(os.path.join(root, 'annotations', file))
        # interpolate the frames
        # annotations = interpolate_annotation(annotations)

        video_path = os.path.join(root, 'videos', file[:-4] + '.mp4')
        save_path = os.path.join(root, 'annotations_txt', file[:-4] + '.txt')

        if not os.path.exists(save_path):
            play_video(save_path, video_path, annotations)


if __name__ == '__main__':
    # root = os.getcwd()[:-11] + '/filtered'  # messy but should do if all named correctly
    root = '/media/hayden/CASR_ACVT/data/filtered'  # todo remove direct path

    aligner(root=root)