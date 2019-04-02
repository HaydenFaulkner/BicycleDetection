# BicycleDetection

## Prerequisites
Python 3.x

## Install

Make two symbolically linked (shortcut) directories `\data` and `\models` within this project. Each will store the data and the models respectively.
```commandline
cd \path\to\BicycleDetection

# From within the BicycleDetection project directory
ln -s \source\with\data data
ln -s \source\with\models models
```
Make another directory `\data\unfiltered` and put the videos and `.saa` annotation files into this directory, if they are in subdirectories that is fine, however make sure all filenames are unique.
Video files are matched with annotation files based on filenames.
Files are filtered into a `\data\filtered` directory using 
## Data Handling

## TODO
When Tensorflow 2.0 official and obj det lib available in 2.0 then might add tf models.
