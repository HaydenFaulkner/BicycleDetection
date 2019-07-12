"""
A CycleDataset to load, store and access the data for training and testing

"""

import math
import numpy as np
import os
import random

import mxnet as mx

from visualisation.image import pil_plot_bbox
from gluoncv.data.base import VisionDataset

from data_processing.annotation import load_annotation_txt_data, interpolate_annotation
from data_processing.image import extract_frames


def _transform_label(label, height=None, width=None):
    # todo modify as not header
    """
    Taken from https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/recordio/detection.py
    """
    label = np.array(label).ravel()
    header_len = int(label[0])  # label header
    label_width = int(label[1])  # the label width for each object, >= 5
    if label_width < 5:
        raise ValueError(
            "Label info for each object should >= 5, given {}".format(label_width))
    min_len = header_len + 5
    if len(label) < min_len:
        raise ValueError(
            "Expected label length >= {}, got {}".format(min_len, len(label)))
    if (len(label) - header_len) % label_width:
        raise ValueError(
            "Broken label of size {}, cannot reshape into (N, {}) "
            "if header length {} is excluded".format(len(label), label_width, header_len))
    gcv_label = label[header_len:].reshape(-1, label_width)
    # swap columns, gluon-cv requires [xmin-ymin-xmax-ymax-id-extra0-extra1-xxx]
    ids = gcv_label[:, 0].copy()
    gcv_label[:, :4] = gcv_label[:, 1:5]
    gcv_label[:, 4] = ids
    # restore to absolute coordinates
    if height is not None:
        gcv_label[:, (0, 2)] *= width
    if width is not None:
        gcv_label[:, (1, 3)] *= height
    return gcv_label


class CycleDataset(VisionDataset):
    """BiCycle dataset.
    Parameters
    ----------
    root : str
        Path to folder storing the dataset.
    """

    def __init__(self, root, split_id, split, categories=['cyclist'], sample_type='frames',
                 cache_frames=False, shuffle=True, percent=1):

        assert sample_type in ['clips', 'frames']

        super(CycleDataset, self).__init__(root)
        self._root = os.path.expanduser(root)
        self._sample_type = sample_type
        self._cache_frames = cache_frames

        # list of the splits to do
        self._split_id = split_id
        self._split = split

        # a tuple with category ids
        # a dictionary keyed with category ids to names
        # a dictionary keyed with category names to ids
        self._classes = categories
        self._categories, self._category_to_names, self._names_to_category = self._load_categories(categories)

        self._data = self._load_data()  # a dictionary containing all the necessary data, keyed on clips with sub dicts keyed on frames

        # samples keyed by their id
        self._samples = self._load_samples()
        self._sample_ids = list(self._samples.keys())
        if percent < 1:
            step = int(math.floor(len(self._sample_ids) / (len(self._sample_ids)*percent)))
            self._sample_ids = [self._sample_ids[i] for i in range(0, len(self._sample_ids), step)]
        if shuffle:
            random.shuffle(self._sample_ids)

    @property
    def classes(self):
        return self._classes

    @property
    def num_categories(self):
        """Number of categories."""
        return len(self._categories)

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, idx):
        label = self.get_boxes(self._sample_ids[idx])
        label = np.array(label)
        img = self._get_frame(self._samples[self._sample_ids[idx]], cache=self._cache_frames)
        img = np.squeeze(img)
        img = mx.nd.array(img, dtype='uint8')

        return img, label

    def _generate_image_path(self, sample):

        if not os.path.exists(os.path.join(self._root, 'frames', sample[0])):
            os.makedirs(os.path.join(self._root, 'frames', sample[0]))
        return os.path.join(self._root, 'frames', sample[0])

    def _get_frame(self, sample, cache=False):
        img_path = None
        if cache:
            img_path = self._generate_image_path(sample)
        video_path = os.path.join(self._root, 'videos', sample[0][:-4]+'.mp4')
        frame = extract_frames(video_path, get_frames=[sample[1]], save_path=img_path)
        return frame

    def _load_data(self):
        data = {}
        for file in os.listdir(os.path.join(self._root, 'annotations_txt')):

            annotation = load_annotation_txt_data(os.path.join(self._root, 'annotations_txt', file))
            # interpolate the frames
            annotation = interpolate_annotation(annotation)

            annotation['video_path'] = os.path.join(self._root, 'videos', file[:-4] + '.mp4')

            # Make sure it's a category we want
            d_keys = set(annotation['instances'].keys())
            for instance_id, instance in annotation['instances'].items():
                for category in self._names_to_category.keys():
                    if category in instance['name']:
                        d_keys.remove(instance_id)
                        for kb in instance['key_boxes'].values():
                            kb.append(self._names_to_category[category])  # add category labels

            for d_key in d_keys:
                del annotation['instances'][d_key]

            # assign to the data dictionary
            data[file] = annotation

        return data

    def _load_samples(self):
        samples = {}
        assert os.path.exists(os.path.join(self._root, 'splits', self._split_id + "_"+self._split+".txt"))

        with open(os.path.join(self._root, 'splits', self._split_id + "_"+self._split+".txt"), 'r') as f:
            lines = [line.rstrip().split('\t') for line in f.readlines()]

        for c, line in enumerate(lines):
            samples[int(line[0])] = (line[1], int(line[2]))

        return samples

    @staticmethod
    def _load_categories(_categories):
        categories = []
        category_to_names = {}
        names_to_category = {}

        for category in _categories:
            category_id = len(categories)
            categories.append(category_id)
            category_to_names[category_id] = category
            names_to_category[category] = category_id

        return categories, category_to_names, names_to_category

    def get_set_boxes(self):
        boxes = {}
        for sample_id in self.ids:
            boxes[sample_id] = self.get_boxes(sample_id)
        return boxes

    def get_boxes(self, sample_id):
        # todo how to handle for clip sample types, ie no frames, return dict or list of lists?
        # todo keep instance data?
        # sample_id = self._sample_ids[index]
        clip_id, frame_id = self._samples[sample_id]
        clip = self._data[clip_id]

        if self._sample_type == 'clips' or int(frame_id) < 0:
            boxes = {}
            for instance_id, instance in clip['instances'].items():
                for frame_n, box in instance['key_boxes'].items():
                    if frame_n in boxes.keys():
                        boxes[frame_n].append(box)
                    else:
                        boxes[frame_n] = [box]
        else:
            boxes = []
            for instance_id, instance in clip['instances'].items():
                if frame_id in instance['key_boxes'].keys():
                    boxes.append(instance['key_boxes'][frame_id])
        return boxes

    def get_set_captions(self):
        captions = {}
        for sample_id in self.ids:
            captions[sample_id] = self.get_captions(sample_id)
        return captions

    def get_category_index_from_name(self, name):
        return self.categories.index(self.names_to_category[name])

    def get_category_id_from_name(self, cid):
        return self.category_to_names[cid]

    def get_category_name_from_index(self, index):
        return self.category_to_names[self.categories[index]]

    def get_category_name_from_id(self, cid):
        return self.category_to_names[cid]


    def load_splits(self, split_id):
        self._sample_ids = []
        self._sample_ids_map = {}

        # Training ids
        train_ids = []
        with open(os.path.join(self._root, 'splits', split_id + "_train.txt"), 'r') as f:
            lines = [line.rstrip().split('\t') for line in f.readlines()]
        for line in lines:
            train_ids.append(int(line[0]))
            self._sample_ids.append(int(line[0]))
            self._sample_ids_map[int(line[0])] = (line[1], line[2])

        # Validation ids
        val_ids = []
        with open(os.path.join(self._root, 'splits', split_id + "_train.txt"), 'r') as f:
            lines = [line.rstrip().split('\t') for line in f.readlines()]
        for line in lines:
            train_ids.append(int(line[0]))
            self._sample_ids.append(int(line[0]))
            self._sample_ids_map[int(line[0])] = (line[1], line[2])

        # Testing ids
        test_ids = []
        with open(os.path.join(self._root, 'splits', split_id + "_train.txt"), 'r') as f:
            lines = [line.rstrip().split('\t') for line in f.readlines()]
        for line in lines:
            train_ids.append(int(line[0]))
            self._sample_ids.append(int(line[0]))
            self._sample_ids_map[int(line[0])] = (line[1], line[2])

        return train_ids, val_ids, test_ids

    def statistics(self):
        boxes_p_cls, boxes_p_img, samples_p_cls = self._category_counts()

        out_str = "# Images: %d\n" \
                  "# Boxes: %d\n" \
                  "# Categories: %d\n" \
                  "Boxes per image (min, avg, max): %d, %d, %d\n" \
                  "Boxes per category (min, avg, max): %d, %d, %d\n\n\n" % \
                  (len(self._sample_ids), sum(boxes_p_img), len(boxes_p_cls),
                   min(boxes_p_img), sum(boxes_p_img) / len(boxes_p_img), max(boxes_p_img),
                   min(boxes_p_cls), sum(boxes_p_cls) / len(boxes_p_cls), max(boxes_p_cls))

        return out_str

    def _category_counts(self):
        # calculate the number of samples per category, and per image
        boxes_p_cls = [0] * self.num_categories
        samples_p_cls = [0] * self.num_categories
        boxes_p_img = []
        for sample_id in self._sample_ids:
            boxes_this_img = 0
            boxes = self.get_boxes(sample_id)
            samples_p_cls_flag = [0] * self.num_categories
            for label in boxes:
                try:
                    boxes_p_cls[self._categories.index(int(label[4]))] += 1
                except ValueError:
                    print()
                boxes_this_img += 1
                if samples_p_cls_flag[self._categories.index(int(label[4]))] == 0:
                    samples_p_cls_flag[self._categories.index(int(label[4]))] = 1
                    samples_p_cls[self._categories.index(int(label[4]))] += 1

            boxes_p_img.append(boxes_this_img)

        return boxes_p_cls, boxes_p_img, samples_p_cls


if __name__ == '__main__':

    dataset = CycleDataset(root='data/filtered', split_id="001", split="val")

    print(dataset.statistics())

    for s in dataset:
        img = s[0].asnumpy()
        bboxes = s[1]
        labels = [bb[4] for bb in bboxes]
        bboxes = [bb[:4] for bb in bboxes]
        vis = pil_plot_bbox(img, bboxes, out_path=None,
                      scores=None, labels=labels, thresh=0.5, class_names=['cyclist'], colors=None, absolute_coordinates=True)
        vis.show()
        print('')
