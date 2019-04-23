"""
A CycleDataset to load, store and access the data for training and testing

"""

import numpy as np
import os
import random

import mxnet as mx
from mxnet import image
from gluoncv.data.base import VisionDataset

from data_processing.annotation import load_annotation_data, interpolate_annotation
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

    def __init__(self, root, split_id, split, categories=['cyclist'], sample_type='frames', pre_save_frames=False):

        assert sample_type in ['clips', 'frames']

        super(CycleDataset, self).__init__(root)
        self._root = os.path.expanduser(root)
        self._sample_type = sample_type
        self._pre_save_frames = pre_save_frames

        # list of the splits to do
        self._split_id = split_id
        self._split = split

        # a tuple with category ids
        # a dictionary keyed with category ids to names
        # a dictionary keyed with category names to ids
        self._categories, self._category_to_names, self._names_to_category = self._load_categories(categories)

        self._data = self._load_data()  # a dictionary containing all the necessary data, keyed on clips with sub dicts keyed on frames

        # samples keyed by their id
        self._samples = self._load_samples()
        self._sample_ids = list(self._samples.keys())
        self._sample_ids = self._sample_ids[:2000]  #todo debug limit set to 500 samples
        random.shuffle(self._sample_ids)


    # def __str__(self):
    #     detail = ','.join([str(s) for s in self.splits])
    #     return self.__class__.__name__ + '(' + detail + ')'

    @property
    def num_categories(self):
        """Number of categories."""
        return len(self._categories)

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, idx):
        label = self.get_boxes(self._sample_ids[idx])
        label = np.array(label)#, dtype=np.float)
        if self._pre_save_frames:
            img_path = self._generate_image_path(self._samples[self._sample_ids[idx]])

            # if self._transform is not None:
            #     img = self._transform(image.imread(img_path), label)
            # img = image.imread(img_path)  # 1 is rgb

        else:
            img = self._get_frame(self._samples[self._sample_ids[idx]])
            # if self._transform is not None:
            #     img = self._transform(image.imdecode(img), label)
            # img = image.imdecode(img)
            img = np.squeeze(img)
            # img = np.swapaxes(img, 0, 2)
            img = mx.nd.array(img, dtype='uint8')
            # todo take from videos directly

        # h, w, _ = img.shape
        # if True:  # self._coord_normalized:
        #     label = _transform_label(label, h, w)
        # else:
        #     label = _transform_label(label)
        return img, label

    def _generate_image_path(self, sample):

        if not os.path.exists(os.path.join(root, 'frames', sample[0])):
            os.makedirs(os.path.join(root, 'frames', sample[0]))
        return os.path.join(root, 'frames', sample[0])

    def _get_frame(self, sample):
        # img_path = self._generate_image_path(sample)
        video_path = os.path.join(self._root, 'videos', sample[0][:-4]+'.mp4')
        frame = extract_frames(video_path, get_frames=[sample[1]], save_path=None)
        return frame

    def _load_data(self):
        data = {}
        for file in os.listdir(os.path.join(self._root, 'annotations')):

            annotation = load_annotation_data(os.path.join(self._root, 'annotations', file))
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


class CycleDatasetAll(object):
    """BiCycle dataset.
    Parameters
    ----------
    root : str
        Path to folder storing the dataset.
    """

    def __init__(self, root, categories=['cyclist'], split_id=None, sample_type='frames', allow_empty=False, shuffle='clips'):
        
        assert sample_type in ['clips', 'frames']
        assert shuffle in [None, 'clips', 'frames']  # No shuffling, shuffle clips (clips exclusive to split), shuffle frames
        if sample_type == 'clips':
            assert shuffle in [None, 'clips']

        super(CycleDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self._sample_type = sample_type
        self._allow_empty = allow_empty

        # list of the splits to do
        self._splits = []

        # a tuple with category ids
        # a dictionary keyed with category ids to names
        # a dictionary keyed with category names to ids
        self._categories, self._category_to_names, self._names_to_category = self._load_categories(categories)

        self._data = self._load_data()  # a dictionary containing all the necessary data, keyed on clips with sub dicts keyed on frames

        # list of sample ids, that are NOT keys to self.data, this can be shuffled to arrange the set order
        # a mapping from sample ids to clips and frames [0] -> (clip_id) or (clip_id, frame_id/num)
        self._sample_ids, self._sample_ids_map = self._load_sample_ids()

        # if split_id not None then this will modify self._sample_ids, self._sample_ids_map
        self.training_ids, self.validation_ids, self.test_ids = self.determine_splits(split_id=split_id)

    # def __str__(self):
    #     detail = ','.join([str(s) for s in self.splits])
    #     return self.__class__.__name__ + '(' + detail + ')'

    @property
    def num_categories(self):
        """Number of categories."""
        return len(self._categories)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return NotImplementedError

    def _load_data(self):
        data = {}
        for file in os.listdir(os.path.join(self._root, 'annotations')):

            annotation = load_annotation_data(os.path.join(self._root, 'annotations', file))
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
                            kb = kb.append(self._names_to_category[category])  # add category labels

            for d_key in d_keys:
                del annotation['instances'][d_key]

            # assign to the data dictionary
            data[file] = annotation

        return data
    
    def _load_sample_ids(self):
        sample_ids = []
        sample_ids_map = {}

        if self._sample_type == 'clips':
            for clip_id, clip in self._data.items():
                sample_id = len(sample_ids)
                sample_ids.append(sample_id)
                sample_ids_map[sample_id] = (clip_id, -1)
        elif self._allow_empty:  # all frames are samples even if they don't contain boxes

            for clip_id, clip in self._data.items():
                for frame_id in range(clip['total_frames']):
                    sample_id = len(sample_ids)
                    sample_ids.append(sample_id)
                    sample_ids_map[sample_id] = (clip_id, frame_id)
        else:
            for clip_id, clip in self._data.items():
                frames = set([])
                for instance_id, instance in clip['instances'].items():
                    for frame_id, box in instance['key_boxes'].items():
                        if frame_id not in frames:
                            if frame_id < 0:
                                print(instance)
                                continue
                            frames.add(frame_id)
                            sample_id = len(sample_ids)
                            sample_ids.append(sample_id)
                            sample_ids_map[sample_id] = (clip_id, frame_id)

        return sample_ids, sample_ids_map

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
        clip_id, frame_id = self._sample_ids_map[sample_id]
        clip = self._data[clip_id]

        if self._sample_type == 'clips' or frame_id < 0:
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

    def determine_splits(self, split_id=None, ratios=[.8, .1, .1], exclusive_clips=True):
        # decided to make a class method since need to know about categories etc..
        assert sum(ratios) == 1

        # Load premade splits from files, instead of making them
        if split_id and os.path.exists(os.path.join(self._root, 'splits', split_id+"_train.txt")):
            return self.load_splits(split_id)

        # we don't have sample_ids here, so use (clip_id, frame_id)
        n_samples = len(self._sample_ids)
        val_start_ind = int(n_samples * ratios[0])
        test_start_ind = int(n_samples * (ratios[0]+ratios[1]))

        # if not keeping clips exclusive can just shuffle all and split
        if not exclusive_clips:  # uncomment if want the split based on frame counts, and self._sample_type == 'frames':
            random.shuffle(self._sample_ids)
            train_ids = self._sample_ids[:val_start_ind]
            val_ids = self._sample_ids[val_start_ind:test_start_ind]
            test_ids = self._sample_ids[test_start_ind:]

        # but if we keep clips exclusive we need to determine the total number of frames for all clips
        else:
            ids_by_clip = {}
            for sample_id in self._sample_ids:
                (clip_id, frame_id) = self._sample_ids_map[sample_id]
                if clip_id in ids_by_clip:
                    ids_by_clip[clip_id].append(sample_id)
                else:
                    ids_by_clip[clip_id] = [sample_id]

            clips_shuffled = list(ids_by_clip.keys())
            random.shuffle(clips_shuffled)

            train_ids, val_ids, test_ids = [], [], []
            count = 0
            for clip_id in clips_shuffled:
                if count < val_start_ind:
                    train_ids += ids_by_clip[clip_id]
                elif count < test_start_ind:
                    val_ids += ids_by_clip[clip_id]
                else:
                    test_ids += ids_by_clip[clip_id]
                count += len(ids_by_clip[clip_id])

        return train_ids, val_ids, test_ids

    def save_splits(self, split_id):
        os.makedirs(os.path.join(self._root, 'splits'), exist_ok=True)
        with open(os.path.join(self._root, 'splits', split_id+"_train.txt"), 'w') as f:
            for sample_id in self.training_ids:
                f.write("%d\t%s\t%s\n" % (sample_id, self._sample_ids_map[sample_id][0], self._sample_ids_map[sample_id][1]))
        with open(os.path.join(self._root, 'splits', split_id+"_val.txt"), 'w') as f:
            for sample_id in self.validation_ids:
                f.write("%d\t%s\t%s\n" % (sample_id, self._sample_ids_map[sample_id][0], self._sample_ids_map[sample_id][1]))
        with open(os.path.join(self._root, 'splits', split_id+"_test.txt"), 'w') as f:
            for sample_id in self.test_ids:
                f.write("%d\t%s\t%s\n" % (sample_id, self._sample_ids_map[sample_id][0], self._sample_ids_map[sample_id][1]))

    def load_splits(self, split_id):
        self._sample_ids = []
        self._sample_ids_map = {}

        # Training ids
        train_ids = []
        with open(os.path.join(self._root, 'splits', split_id+"_train.txt"), 'r') as f:
            lines = [line.rstrip().split('\t') for line in f.readlines()]
        for line in lines:
            train_ids.append(int(line[0]))
            self._sample_ids.append(int(line[0]))
            self._sample_ids_map[int(line[0])] = (line[1], line[2])

        # Validation ids
        val_ids = []
        with open(os.path.join(self._root, 'splits', split_id+"_train.txt"), 'r') as f:
            lines = [line.rstrip().split('\t') for line in f.readlines()]
        for line in lines:
            train_ids.append(int(line[0]))
            self._sample_ids.append(int(line[0]))
            self._sample_ids_map[int(line[0])] = (line[1], line[2])

        # Testing ids
        test_ids = []
        with open(os.path.join(self._root, 'splits', split_id+"_train.txt"), 'r') as f:
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
    # root = os.getcwd()[:-11] + '/filtered'  # messy but should do if all named correctly
    root = '/media/hayden/CASR_ACVT/data/filtered'  # todo remove direct path

    dataset = CycleDataset(root=root, split_id="001", split="val")
    # dataset.save_splits('testing')
    # boxes = dataset.get_boxes(1)

    print(dataset.statistics())

    from visualisation.image import pil_plot_bbox
    for s in dataset:
        img = s[0].asnumpy()
        bboxes = s[1]
        labels = [bb[4] for bb in bboxes]
        bboxes = [bb[:4] for bb in bboxes]
        vis = pil_plot_bbox(img, bboxes, out_path=None,
                      scores=None, labels=labels, thresh=0.5, class_names=['cyclist'], colors=None, absolute_coordinates=True)
        vis.show()
        print('')
