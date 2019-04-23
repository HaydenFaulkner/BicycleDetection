"""
For loading datasets and building splits with the data
"""

import os
import random

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform

from data_processing.annotation import load_annotation_data, interpolate_annotation
from data_processing.dataset import CycleDataset


def load_data(root, categories):
    data = {}
    for file in os.listdir(os.path.join(root, 'annotations')):

        annotation = load_annotation_data(os.path.join(root, 'annotations', file))
        # interpolate the frames
        annotation = interpolate_annotation(annotation)

        annotation['video_path'] = os.path.join(root, 'videos', file[:-4] + '.mp4')

        # Make sure it's a category we want
        d_keys = set(annotation['instances'].keys())
        for instance_id, instance in annotation['instances'].items():
            for category in categories:
                if category in instance['name']:
                    d_keys.remove(instance_id)
                    for kb in instance['key_boxes'].values():
                        kb.append(categories.index(category))  # add category labels

        for d_key in d_keys:
            del annotation['instances'][d_key]

        # assign to the data dictionary
        data[file] = annotation

    return data


def load_sample_ids(data, sample_type='frames', allow_empty=False):
    sample_ids = {}

    if sample_type == 'clips':
        for clip_id, clip in data.items():
            sample_id = len(sample_ids)
            sample_ids[sample_id] = (clip_id, -1)
    elif allow_empty:  # all frames are samples even if they don't contain boxes

        for clip_id, clip in data.items():
            for frame_id in range(clip['total_frames']):
                sample_id = len(sample_ids)
                sample_ids[sample_id] = (clip_id, frame_id)
    else:
        for clip_id, clip in data.items():
            frames = set([])
            for instance_id, instance in clip['instances'].items():
                for frame_id, box in instance['key_boxes'].items():
                    if frame_id not in frames:
                        if frame_id < 0:
                            print(instance)
                            continue
                        frames.add(frame_id)
                        sample_id = len(sample_ids)
                        sample_ids[sample_id] = (clip_id, frame_id)

    return sample_ids


def save_split(root, split_id, sample_ids, split='train'):

    assert split in ['train', 'val', 'test']

    os.makedirs(os.path.join(root, 'splits'), exist_ok=True)
    with open(os.path.join(root, 'splits', split_id+"_"+split+".txt"), 'w') as f:
        for sample_id in sample_ids:
            f.write("%d\t%s\t%s\n" % (sample_id, sample_ids[sample_id][0], sample_ids[sample_id][1]))


def load_split(root, split_id, split='train'):

    assert split in ['train', 'val', 'test']

    sample_ids = {}

    with open(os.path.join(root, 'splits', split_id+"_"+split+".txt"), 'r') as f:
        lines = [line.rstrip().split('\t') for line in f.readlines()]
    for line in lines:
        sample_ids[int(line[0])] = (line[1], line[2])

    return sample_ids


def generate_splits(root, split_id, sample_ids, ratios=[.8, .1, .1], exclusive_clips=True, save=True):
    # decided to make a class method since need to know about categories etc..
    assert sum(ratios) == 1

    # Load premade splits from files, instead of making them
    if split_id and \
            os.path.exists(os.path.join(root, "splits", split_id + "_train.txt")) and \
            os.path.exists(os.path.join(root, "splits", split_id + "_val.txt")) and \
            os.path.exists(os.path.join(root, "splits", split_id + "_test.txt")):
        return load_split(root, split_id, split="train"), \
               load_split(root, split_id, split="val"), \
               load_split(root, split_id, split="test")

    # we don't have sample_ids here, so use (clip_id, frame_id)
    n_samples = len(sample_ids)
    val_start_ind = int(n_samples * ratios[0])
    test_start_ind = int(n_samples * (ratios[0] + ratios[1]))

    # if not keeping clips exclusive can just shuffle all and split
    if not exclusive_clips:  # uncomment if want the split based on frame counts, and sample_type == 'frames':
        random.shuffle(sample_ids)
        train_ids = sample_ids[:val_start_ind]
        val_ids = sample_ids[val_start_ind:test_start_ind]
        test_ids = sample_ids[test_start_ind:]

    # but if we keep clips exclusive we need to determine the total number of frames for all clips
    else:
        ids_by_clip = {}
        for sample_id in sample_ids:
            (clip_id, frame_id) = sample_ids[sample_id]
            if clip_id in ids_by_clip:
                ids_by_clip[clip_id].append(sample_id)
            else:
                ids_by_clip[clip_id] = [sample_id]

        clips_shuffled = list(ids_by_clip.keys())
        random.shuffle(clips_shuffled)

        train_ids, val_ids, test_ids = {}, {}, {}
        count = 0
        for clip_id in clips_shuffled:
            for sample_id in ids_by_clip[clip_id]:

                if count < val_start_ind:
                    train_ids[sample_id] = sample_ids[sample_id]
                elif count < test_start_ind:
                    val_ids[sample_id] = sample_ids[sample_id]
                else:
                    test_ids[sample_id] = sample_ids[sample_id]
            count += len(ids_by_clip[clip_id])  # this ensure a clip isnt split across sets

    if save:
        save_split(root, split_id, sample_ids=train_ids, split='train')
        save_split(root, split_id, sample_ids=val_ids, split='val')
        save_split(root, split_id, sample_ids=test_ids, split='test')

    return train_ids, val_ids, test_ids


def load_datasets(root, split_id, categories):
    if not os.path.exists(os.path.join(root, 'splits', split_id + "_train.txt")) or \
       not os.path.exists(os.path.join(root, 'splits', split_id + "_val.txt")) or \
       not os.path.exists(os.path.join(root, 'splits', split_id + "_test.txt")):
        data = load_data(root, categories)
        sample_ids = load_sample_ids(data, sample_type='frames', allow_empty=False)
        generate_splits(root, split_id=split_id, sample_ids=sample_ids, ratios=[.8, .1, .1],
                        exclusive_clips=True, save=True)

    train_dataset = CycleDataset(root=root, split_id=split_id, split="train")
    val_dataset = CycleDataset(root=root, split_id=split_id, split="val")
    test_dataset = CycleDataset(root=root, split_id=split_id, split="test")

    return train_dataset, val_dataset, test_dataset


def get_dataloader(net, train_dataset, val_dataset, test_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    val_loader = None
    if val_dataset:
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(SSDDefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)

    test_loader = None
    if test_dataset:
        test_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        test_loader = gluon.data.DataLoader(
            test_dataset.transform(SSDDefaultValTransform(width, height)),
            batch_size, False, batchify_fn=test_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # root = os.getcwd()[:-11] + '/filtered'  # messy but should do if all named correctly
    root = '/media/hayden/CASR_ACVT/data/filtered'  # todo remove direct path

    data = load_data(root, ['cyclist'])
    sample_ids = load_sample_ids(data, sample_type='frames', allow_empty=False)
    generate_splits(root, split_id="001", sample_ids=sample_ids, ratios=[.8, .1, .1], exclusive_clips=True, save=True)

    train_dataset, val_dataset, test_dataset = load_datasets(root, split_id="001")
    print('h')
