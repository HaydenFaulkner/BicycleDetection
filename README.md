# BicycleDetection


## Install

Make two symbolically linked (shortcut) directories `data` and `models` within this project. Each will store the data and the models respectively.
```commandline
cd path/to/BicycleDetection

# From within the BicycleDetection project directory
ln -s source/with/data data
ln -s source/with/models models
```
## Training
You can skip this section
### Data processing
Make another directory `data/unfiltered` and put the videos and `.saa` annotation files into this directory, if they are in subdirectories that is fine, however make sure all filenames are unique.
Video files are matched with annotation files based on filenames.
Files are filtered into a `data/filtered` directory using 

Will need to use aligner to align `.saa` files

Use `train_yolo.py` in the `run` directory

## Usage
In the main directory there are 6 python scripts available to run to perform different things:
- `video_to_frames.py` is used to extract frames from a video, the frames are used in the detector and are loaded more
efficiently than getting them directly from the video;
- `detect.py` is used to detect cyclists in the images using a fast YOLOv3 + MobileNet1.0 object detection framework.
This script will output a text file with the detections in the form: `frame_num,class_id,confidence,left,top,right,bottom`;
- `track.py` is used to associate the detections into tracks and will similarly output a text file with the tracks in
the form: `frame_num,track_id,confidence,left,top,right,bottom`;
- `visualise.py` is used to visualise tracks and detections on the original video. It can also be used to snapshot 
individual cyclists as clips and/or cropped box images. It can also produce an overlay of all tracks in a video in a still;
- `full.py` runs all of the above scripts across a set of videos held in `data/unprocessed` from videos to visualisation.
You can do per video or per process/script using the flags (I'm yet to test which is faster);
- `subclipper.py` will take videos held in `data/to_shorted` and output shortened clips to `data\clips`, performing
frame extraction, detection, and tracking if necessary.

You can check the available options for each script using the `--help` flag:
```commandline

path/to/BicycleDetection$ python full.py --help
```

You can change any parameter by using its flag and then the new value as so:
```commandline

path/to/BicycleDetection$ python subclipper.py --start_buffer 10 --display_tracks True
```