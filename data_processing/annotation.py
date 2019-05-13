"""
data_processing.annotation contains functions for processing annotation data

"""
import copy
import numpy as np
import os
import sqlite3
import subprocess


def organise_data():
    """
    gets all annotation files and video files from root/unfiltered and puts them in root/filtered/annotations and
    root/filtered/videos respectively, it also removes any which don't match

    :param root: str, the root path
    :return: int_list, the list of common files that we keep
    """
    # Get the root of the project, and then the data dir
    root = os.path.dirname(os.path.abspath(__file__))
    root = root[:root.rfind('BicycleDetection')+len('BicycleDetection')+1]
    root = os.path.join(root, 'data')
    root = '/media/hayden/CASR_ACVT/data' ##################################################### TODO REMOVE BEFORE HANDOVER
    print("Find and copy all the annotation files")
    os.makedirs(os.path.join(root, 'filtered', 'annotations'))
    cmd = "find " + root + "/unfiltered/. -name \*.saa -exec cp -v {} " + root + "/filtered/annotations/ \;"
    process = subprocess.call(cmd, shell=True)

    print("Find and copy all the video files")
    os.makedirs(os.path.join(root, 'filtered', 'videos'))
    cmd = "find " + root + "/unfiltered/. -name \*.mp4 -exec cp -v {} " + root + "/filtered/videos/ \;"
    process = subprocess.call(cmd, shell=True)

    print("Removing files that don't have corresponding annotations or videos")
    ann_list = os.listdir(os.path.join(root, 'filtered', 'annotations'))
    ann_list = [ann[:-4] for ann in ann_list]
    vid_list = os.listdir(os.path.join(root, 'filtered', 'videos'))
    vid_list = [vid[:-4] for vid in vid_list]
    int_list = [value for value in ann_list if value in vid_list]

    for vid_file in vid_list:
        if vid_file not in int_list:
            os.remove(os.path.join(root, 'filtered', 'videos', vid_file + '.mp4'))
            print('Removing: ' + os.path.join(root, 'filtered', 'videos', vid_file + '.mp4'))

    for ann_file in ann_list:
        if ann_file not in int_list:
            os.remove(os.path.join(root, 'filtered', 'annotations', ann_file + '.saa'))
            print('Removing: ' + os.path.join(root, 'filtered', 'annotations', ann_file + '.saa'))

    return int_list


def load_annotation_data(saa_file_path):
    """
    Loads data from a single .saa annotation file (sqlite3 db file) into a dictionary

    :param saa_file_path: str, the path to the .saa file
    :return annotation: dict, containing all the important info
    """
    if not os.path.exists(saa_file_path):
        raise FileNotFoundError

    try:
        conn = sqlite3.connect(saa_file_path)
    except sqlite3.Error as e:
        print(e)
        return None

    cur = conn.cursor()

    annotation = {}

    # Grab the video data
    cur.execute("SELECT * FROM VIDEO")
    rows = cur.fetchall()
    assert len(rows) == 1
    # for row in rows:
    #     print(row)
    annotation['video_path'] = rows[0][0]
    annotation['video_file'] = rows[0][1]
    annotation['total_frames'] = rows[0][2]
    annotation['width'] = rows[0][3]
    annotation['height'] = rows[0][4]
    annotation['fps'] = rows[0][6]

    # Grab the object instance ids
    annotation['instances'] = {}
    cur.execute("SELECT * FROM VO")
    rows = cur.fetchall()
    for row in rows:
        annotation['instances'][row[0]] = {'name': row[1], 'key_boxes': {}}

    # Load the boxes from the key frames
    cur.execute("SELECT * FROM VOP")
    rows = cur.fetchall()
    for row in rows:
        # get box coords
        coords = row[4].split(',')
        coords_x = []
        coords_y = []

        # just make a bounding box around the vertices
        for i, coord in enumerate(coords):
            if i % 2 == 0:
                coords_x += [int(coord)]
            else:
                coords_y += [int(coord)]

        # what key frame is this
        if row[5] != '1':
            lrm = 1   # final key frame
        elif row[6] != '1':
            lrm = -1  # start key frame
        else:
            lrm = 0
        # for some reason all the key frames are 12 frames off...fix in display!?! causes errors
        # annotation['instances'][row[1]]['key_boxes'][row[0]-12] = [min(coords_x), min(coords_y),
        #                                                            max(coords_x), max(coords_y),
        #                                                            lrm]
        annotation['instances'][row[1]]['key_boxes'][row[0]] = [min(coords_x), min(coords_y),
                                                                max(coords_x), max(coords_y),
                                                                lrm]

    return annotation


def load_annotation_txt_data(txt_file_path):
    """
    Loads data from a single .txt annotation file into a dictionary

    :param txt_file_path: str, the path to the .txt file
    :return annotation: dict, containing all the important info
    """
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError

    annotation = dict()

    # Grab the object instance ids
    annotation['instances'] = dict()

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip().split(',') for line in lines]

    for line in lines:
        if line[6] not in annotation['instances']:
            annotation['instances'][line[6]] = {'name': line[6], 'key_boxes': {}}

        annotation['instances'][line[6]]['key_boxes'][int(line[0])] = [int(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5])]

    return annotation


def interpolate_annotation(annotation):
    annotation = copy.deepcopy(annotation)

    for k, instance in annotation['instances'].items():  # for each instances
        frames = list(instance['key_boxes'].keys())
        frames.sort()

        for i in range(len(frames)-1):
            start_box = instance['key_boxes'][frames[i]]
            end_box = instance['key_boxes'][frames[i+1]]

            # if the flags are right we want to interp between these two frames
            if start_box[4] != 1 and end_box[4] != -1:
                boxes = interpolate_boxes(frames[i], frames[i+1], start_box[:4], end_box[:4])

                for f, box in boxes.items():
                    instance['key_boxes'][f] = box

            if start_box[4] == 1 or start_box[4] == -1:
                instance['key_boxes'][frames[i]] = start_box[:4]
            if i == len(frames) - 2:
                instance['key_boxes'][frames[i+1]] = end_box[:4]

    return annotation


def interpolate_boxes(start_frame, end_frame, start_box, end_box):
    frames = list(range(start_frame, end_frame))
    boxes = {}
    for i in range(len(frames)):
        perc = float(i)/float(len(frames))

        x_min = start_box[0] + int(np.round(perc*(end_box[0]-start_box[0])))
        y_min = start_box[1] + int(np.round(perc*(end_box[1]-start_box[1])))
        x_max = start_box[2] + int(np.round(perc*(end_box[2]-start_box[2])))
        y_max = start_box[3] + int(np.round(perc*(end_box[3]-start_box[3])))
        boxes[frames[i]] = [x_min, y_min, x_max, y_max]

    return boxes


if __name__ == '__main__':

    organise_data()

    # for file in os.listdir('/media/hayden/CASR_ACVT/annotations'):
    #     annotation = load_annotation_data(os.path.join('/media/hayden/CASR_ACVT/annotations', file))

    # annotation = load_annotation_data('/media/hayden/CASR_ACVT/annotations/R021.saa')
    # annotation = interpolate_annotation(annotation)
    # print('')