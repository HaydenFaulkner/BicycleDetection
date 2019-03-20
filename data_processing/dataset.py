"""
A CycleDataset to load, store and access the data for training and testing

"""

import os
import random

from data_processing.annotation import load_annotation_data, interpolate_annotation


class CycleDataset(object):
    """BiCycle dataset.
    Parameters
    ----------
    root : str
        Path to folder storing the dataset.
    """

    def __init__(self, root, categories=['cyclist'], split=None, sample_type='frames', allow_empty=False, shuffle='clips'):
        
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

        self.training_ids, self.validation_ids, self.test_ids = self.determine_splits()

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
            categories.append(category)
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
            for instance_id, instance in clip['instances']:
                for frame_n, box in instance['key_boxes']:
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

    def statistics(self):
        return ""

    def determine_splits(self, ratios=[.8, .1, .1], exclusive_clips=True):
        # decided to make a class method since need to know about categories etc..
        assert sum(ratios) == 1

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

if __name__ == '__main__':
    root = '/media/hayden/CASR_ACVT/'

    dataset = CycleDataset(root=root)
    boxes = dataset.get_boxes(0)

    print()
