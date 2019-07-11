"""
Used to process a directory of videos, subclipping around all cyclist appearances

"""

import argparse
import os
import queue
import time

from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms import bbox as tbbox
import mxnet as mx
import numpy as np
import cv2

from run.evaluation.common import transform_test
from visualisation.image import cv_plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Process a directory of videos, subclipping around all cyclist appearances')
    parser.add_argument('--dir', type=str, default="data/vids_to_process",
                        help="Directory path")
    parser.add_argument('--model', type=str, default="models/002_faster_rcnn_resnet50_v1b_custom_cycle/best.params",
                        help="Model path")
    parser.add_argument('--every', type=int, default=25,
                        help="Detect every this many frames. Default is 25.")
    parser.add_argument('--boxes', type=bool, default=True,
                        help="Display bounding boxes on the processed frames.")
    parser.add_argument('--gpus', type=str, default="0",
                        help="GPU ids to use, defaults to '0', if want CPU set to ''. Use comma for multiple eg. '0,1'.")
    parser.add_argument('--buffer', type=int, default=25,
                        help="How many frames to buffer around cyclists. Default is 25.")
    parser.add_argument('--separate', type=int, default=0,
                        help="0: only make single summary clip; 1: make both summary and sub clips; 2: make only sub clips. Default is 0.")
    parser.add_argument('--threshold', type=float, default=0.99,
                        help="Threshold on detection confidence. Default is 0.99")

    # parser.add_argument('--backend', type=str, default="mx",
    #                     help="The backend to use: mxnet (mx) or tensorflow (tf). Currently only supports mxnet.")

    return parser.parse_args()


def load_net(model_path, ctx):

    # setup network
    net_id = 'faster_rcnn_resnet50_v1b_custom'
    net = get_model(net_id, pretrained_base=True, classes=['cyclist'])  # just finetuning

    net.load_parameters(model_path)

    net.collect_params().reset_ctx(ctx)

    net.hybridize(static_alloc=True)

    return net


def process_frame(image, net, ctx):
    # currently only supports batch size 1 todo
    image = np.squeeze(image)
    image = mx.nd.array(image, dtype='uint8')
    x, _ = transform_test(image, 600, max_size=1000)
    x = x.copyto(ctx[0])

    # get prediction results
    ids, scores, bboxes = net(x)
    oh, ow, _ = image.shape
    _, _, ih, iw = x.shape
    bboxes[0] = tbbox.resize(bboxes[0], in_size=(iw, ih), out_size=(ow, oh))
    return bboxes[0].asnumpy(), scores[0].asnumpy(), ids[0].asnumpy()


def clip_video(video_dir, clip_dir, video_file, net, ctx, every=25, buffer=25, boxes=False, separate=0, threshold=0.5):
    video_path = os.path.join(video_dir, video_file)
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

    # if ext == '.mp4':
    clip_count = 0

    buffer = max(buffer, every) # ensure the buffer encompasses the window of checking
    current = 0
    buf = queue.Queue(maxsize=buffer)
    writing = False
    last_found = 0
    out = 0
    clip = None
    summary_clip = None
    while True:
        if current % int(total*.1) == 0:
            print("%d%% (%d/%d)" % (int(100*current/total)+1, current, total))

        flag, frame = capture.read()
        if flag == 0 and current < total-2:
            # print("frame %d error flag" % current)
            current += 1
            continue
            #break
        if frame is None:
            break
        height, width, _ = frame.shape

        if current % every == 0:
            bboxes, scores, ids = process_frame(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), net=net, ctx=ctx)

            if scores[0] > threshold:  # we found a box
                last_found = current
                if boxes:
                    frame = cv_plot_bbox(out_path=None,
                                          img=frame,
                                          bboxes=bboxes,
                                          scores=scores,
                                          labels=ids,
                                          thresh=threshold,
                                          class_names=['cyclist'])
                # found a new clip
                if not writing:
                    if summary_clip is None and separate < 2:
                        print("Making Summary Clip: %s_summary.mp4" % os.path.join(clip_dir, video_file[:-4]))
                        summary_clip = cv2.VideoWriter("%s_summary.mp4" % os.path.join(clip_dir, video_file[:-4]),
                                                       cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

                    # start the clip
                    if separate > 0:
                        print("Making Subclip: %s_%05d.mp4" % (os.path.join(clip_dir, video_file[:-4]), clip_count+1))
                        clip = cv2.VideoWriter("%s_%05d.mp4" % (os.path.join(clip_dir, video_file[:-4]), clip_count+1),
                                               cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))
                    # clip = cv2.VideoWriter("%s_%05d.avi" % (os.path.join(clip_dir, video_file[:-4]), clip_count+1),
                    #                        cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (width, height))

                    # write out the buffer frames pre finding box
                    while not buf.empty():
                        buf_img = buf.get()
                        if separate > 0:
                            clip.write(buf_img)
                        if separate < 2:
                            summary_clip.write(buf_img)

                        out += 1

                    writing = True

        if writing:
            # write out current frame
            if separate > 0:
                clip.write(frame)
            if separate < 2:
                summary_clip.write(frame)
            out += 1

            # if we haven't found one within the buffer stop writing and close clip
            if current - last_found > buffer:
                writing = False
                clip_count += 1
                if clip is not None:
                    clip.release()

        else:
            # put into buffer
            if buf.full():
                buf.get()
            buf.put(frame)

        current += 1

    if clip is not None:
        clip.release()
    if summary_clip is not None:
        summary_clip.release()

    if clip_count < 1:
        print("No Cyclists detected")
    return [out, total]


def subclipper(video_dir, model_path, every=25, gpus='', buffer=25, boxes=False, separate=0, threshold=0.5):
    file_types = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
    # Ensure we are in the BicycleDetection working directory
    if not os.getcwd()[-16:] == 'BicycleDetection':
        print("ERROR: Please ensure 'BicycleDetection' is the working directory")
        return None

    gutils.random.seed(233)

    # contexts ie. gpu or cpu
    ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net = load_net(model_path=model_path, ctx=ctx)

    clip_dir = os.path.join(video_dir, 'sub_clips')
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(os.path.join(video_dir, 'processed'), exist_ok=True)

    out_total = 0
    total_total = 0
    start_time = time.time()
    total_vids_done = 0
    vids = os.listdir(os.path.join(video_dir, 'unprocessed', gpus))
    total_vids = len(vids)
    for i, video_file in enumerate(vids):
        t = time.time()
        if video_file[-4:] not in file_types:
            print('File type %s not supported' % video_file[-4:])
            continue
        out, total = clip_video(video_dir=os.path.join(video_dir, 'unprocessed', gpus),
                                clip_dir=clip_dir, video_file=video_file, net=net, ctx=ctx, every=every, buffer=buffer,
                                boxes=boxes, separate=separate, threshold=threshold)


        # move video to processed dir
        if out > 0:
            total_vids_done += 1
            out_total += out
            total_total += total
            os.rename(os.path.join(video_dir, 'unprocessed', gpus, video_file), os.path.join(video_dir, 'processed', video_file))
        else:
            print("No detections found in video, consider lowering the threshold.")

        print("Processing Video %d of %d Complete (%s). Cut out %d minutes (%0.2f%%). Took %d minutes." % (i+1,
                                                                                                          total_vids,
                                                                                                          video_file,
                                                                                                          int(out/25.0/60.0),
                                                                                                          100-(100*out/(total+.001)),
                                                                                                          int((time.time() - t)/60.0)))

    print("Processed %d of %d Videos. Cut out %d minutes (%0.2f%%). Took %d minutes." % (total_vids_done,
                                                                                         total_vids,
                                                                                         int(out_total/25.0/60.0),
                                                                                         100-(100*out_total/(total_total+.001)),
                                                                                         int((time.time() - start_time)/60.0)))

if __name__ == '__main__':

    args = parse_args()
    subclipper(video_dir=args.dir,
               model_path=args.model,
               every=args.every,
               gpus=args.gpus,
               buffer=args.buffer,
               boxes=args.boxes,
               separate=args.separate,
               threshold=args.threshold)