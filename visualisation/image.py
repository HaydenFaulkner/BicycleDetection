import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

from data_processing.image import extract_frames
from data_processing.annotation import load_annotation_data, interpolate_annotation


def pil_plot_bbox(out_path, img, bboxes, scores=None, labels=None, thresh=0.5, class_names=None, colors=None, absolute_coordinates=True):
    """
    plot bounding boxes on an image and
    """


    if isinstance(img, str):
        img = Image.open(img).convert('RGBA')
    else:
        img = Image.fromarray(img).convert('RGBA')

    if len(bboxes) < 1:
        img.save(out_path, "png")
        return

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height


    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores[i] < thresh:
            continue
        if labels is not None and labels[i] < 0:
            continue
        cls_id = int(labels[i]) if labels is not None else -1
        if cls_id not in colors:
            colors[cls_id] = (int(256*random.random()), int(256*random.random()), int(256*random.random()))
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(255,0,0,60), outline=(255,0,0,255))

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores[i]) if scores is not None else ''
        if class_name or score:

            draw.text((xmin, ymin - 2), '{:s} {:s}'.format(class_name, score), fill=(255,0,0,255))

    img = Image.alpha_composite(img, overlay)
    img.save(out_path, "png")


def display_gt_boxes(video_path, annotation_path, save_path, buffer_length=200, interpolate=True):

    annotations = load_annotation_data(annotation_path)
    if interpolate:
        annotations = interpolate_annotation(annotations)

    buffer_count = 0
    buffer_boxes = []
    for frame_num in tqdm(range(annotations['total_frames'])):
        boxes = []
        for id, instance in annotations['instances'].items():
            if frame_num in instance['key_boxes'].keys():
                boxes.append(instance['key_boxes'][frame_num])

        buffer_boxes.append(boxes)

        if frame_num % buffer_length == 0 or frame_num == annotations['total_frames'] - 1:

            frames = extract_frames(video_path, get_frames=list(range(frame_num+1-len(buffer_boxes), frame_num+1)))

            for buffer_idx in range(len(buffer_boxes)):
                frame = frames[buffer_idx]
                boxes = buffer_boxes[buffer_idx]

                if len(boxes) > 0:
                    pil_plot_bbox(out_path=os.path.join(save_path, "%08d.png" % (frame_num-len(buffer_boxes)+buffer_idx+1)), img=frame, bboxes=boxes,
                                  scores=[1]*len(boxes), labels=[0]*len(boxes), class_names=['bike'])

            buffer_boxes = []
        buffer_count += 1


if __name__ == '__main__':

    root = '/media/hayden/CASR_ACVT/'
    for file in os.listdir(os.path.join(root, 'annotations')):
        annotation_path = os.path.join(root, 'annotations', file)
        video_path = os.path.join(root, 'videos', file[:-4] + '.mp4')
        save_path = os.path.join(root, 'visualisations', file[:-4])
        os.makedirs(save_path, exist_ok=True)

        display_gt_boxes(video_path, annotation_path, save_path)