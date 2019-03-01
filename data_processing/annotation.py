"""
data_processing.annotation contains functions for processing annotation data

"""
import os
import sqlite3
import subprocess


def organise_data(root):
    """
    gets all annotation files and video files from root and puts them in root/annotations and root/videos respectively
    it also removes any which don't match

    :param root: str, the root path
    :return: int_list, the list of common files that we keep
    """
    print("Find and copy all the annotation files")
    os.makedirs(os.path.join(root, 'annotations'))
    cmd = "find " + root + ". -name \*.saa -exec cp -v {} " + root + "annotations/ \;"
    process = subprocess.call(cmd, shell=True)

    print("Find and copy all the video files")
    os.makedirs(os.path.join(root, 'videos'))
    cmd = "find " + root + ". -name \*.mp4 -exec cp -v {} " + root + "videos/ \;"
    process = subprocess.call(cmd, shell=True)

    print("Removing files that don't have corresponding annotations or videos")
    ann_list = os.listdir(os.path.join(root, 'annotations'))
    ann_list = [ann[:-4] for ann in ann_list]
    vid_list = os.listdir(os.path.join(root, 'videos'))
    vid_list = [vid[:-4] for vid in vid_list]
    int_list = [value for value in ann_list if value in vid_list]

    for vid_file in vid_list:
        if vid_file not in int_list:
            os.remove(os.path.join(root, 'videos', vid_file + '.mp4'))
            print('Removing: ' + os.path.join(root, 'videos', vid_file + '.mp4'))

    for ann_file in ann_list:
        if ann_file not in int_list:
            os.remove(os.path.join(root, 'annotations', ann_file + '.saa'))
            print('Removing: ' + os.path.join(root, 'annotations', ann_file + '.saa'))

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

        annotation['instances'][row[1]]['key_boxes'][row[0]] = [min(coords_x), min(coords_y),
                                                                max(coords_x), max(coords_y)]

    return annotation


if __name__ == '__main__':

    # organise_data('/media/hayden/CASR_ACVT/')

    for file in os.listdir('/media/hayden/CASR_ACVT/annotations'):
        annotation = load_annotation_data(os.path.join('/media/hayden/CASR_ACVT/annotations', file))
    print('')
