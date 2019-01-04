# Pedestrian Intention Prediction

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Creating the dataset](#usage)
  * [Training](#usage)
  * [Results](#results)
    * [Prediction](#long-term-prediction)
    * [Long-Term Prediction](#long-term-prediction)
    * [Out of Domain Runs (1 input object)](#out-of-domain-runs-1-input-object)
    * [Out of Domain Runs (3 input objects)](#out-of-domain-runs-3-input-objects)
    * [Generation Diversity](#generation-diversity)

# Requirements
------------
What we used to run the experiments

  * Python 2.7.3
  * Tensorflow 1.3.0
  * Ubuntu 14.04
  * NVIDIA GTX 780M

# Brief Project Structure
------------

    ├── dataset  
        ├── all
            ├── raw                    : folder containing the raw recordings  
            ├── processed              : folder containing the cropped/annotated recordings
            ├── annotations            : folder containing the (approximate) ground truth text file i.e. the output of mask rcnn
            ├── crops                  : folder containing the pedestrian crops
        ├── train                      : folder containing the training set
            ├── annotations
            ├── crops
        ├── test                       : folder containing the testing set
            ├── annotations
            ├── crops
    
    ├── annotator                      
        ├── mask-rcnn.pytorch          : folder containing the mask rcnn people detector 
        ├── hungarian_tracker.py       : python script that runs the hungarian tracker 
        ├── assign_labels.py           : python script that assigns the label (cross / did not cross) to each pedestrian
        ├── crop_images.py             : python script that crops the pedestrian and places them in /dataset/crops
        ├── annotate.sh                : shell script that runs the full annotation pipeline
        ├── annotate-sample1.sh        : shell script that sets the parameters for the sample1 video
     
    ├── intention_prediction
        ├── models                     : pretrained model weights
        ├── results                    : 
        ├── scripts                    : folder containing the model and data loader python scripts
        ├── train.py                   : train script
        ├── train.sh                   : train script
        ├── evaluate_lausanne.ipynb    : ipython script to visualize results on the lausanne dataset
        ├── guidedbp_lausanne.ipynb    : ipython script to visualize guided-backprop results on the lausanne dataset
     
    ├── images                         : images used for this github repository
    ├── report.pdf                     : report
    ├── slides_midterm.pptx            : midterm presentation slides
    ├── slides_final.pptx              : final presentation slides

# Creating the dataset
------------

To augment the dataset of [1], we have that can be automatically annotated. In the following, we describe the steps needed to generate (approximate) the ground truths of your own recordings. The example below is done on the video `/dataset/all/raw/sample1.MP4`. A shell script including all the steps have been included.

1) Follow the instructions at https://github.com/roytseng-tw/Detectron.pytorch to set up the Mask RCNN.

2)	Set the region of interest that the annotator will operate on in the following order (top left x coordinate, top left y coordinate, width, height). The values for the example below is (0, 875, 650, 250). Use ffmpeg to crop the video and place the output in `/dataset/all/processed/sample1.MP4`.

  ```bash
  ffmpeg -y -i ROOT/dataset/all/raw/sample1.MP4 -filter:v "crop=0:875:650:250" ROOT/dataset/all/processed/sample1.MP4
  ```

![Alt Text](/images/dataset/step1.png)

3)	Run the Mask RCNN people detector on the input video at `/dataset/all/processed/sample1.MP4`. The output is a csv text file at  `/dataset/all/annotations/sample1.txt` containing the detections. Note that the UID of each detection is initialized as -1 and will be given a unique UID when the tracker is run.
  
  ```bash
  cd ROOT/mask-rcnn.pytorch
  python3 -u tools/infer_simple.py --dataset keypoints_coco2017 --cfg ./configs/baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml -
  load_detectron ./data/weights/R-50-FPN-1x.pkl --image_dir ../../dataset/all/raw/sample1.MP4 --output_dir
  ../../dataset/all/annotations/sample1.txt
  ``` 
  
  ```python
  df = df.read_csv("sample1.txt")
  print(df)
  ```
| Frame no  | UID | tlx | tlx | width | height | score |  
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | -1 | 474 | 12 | 20 | 56 | 0.995529 |
| 1 | -1 | 474 | 12 | 20 | 56 | 0.995529 |

4)	Run `hungarian.py` and set the `maximum_allowed_distance` (pixels) and `maximum_limbo_lifetime` (frames) parameters. The `maximum_allowed_distance` prevents a detection at `t1` from being assigned to a detection at `t2` if their distance is above said parameter. The `maximum_limbo_lifetime` stops the tracker for any object that has remained in limbo without a successful match i.e. it stops looking for a correspondence for an object if that object has not found a match after said duration. Running `hungarian.py` appends a new column `label` to the dataframe and saves the csv file to `/dataset/all/annotations/sample1-modified.txt`

  ```bash
  cd ROOT
  python3 ./annotator/hungarian.py ../dataset/all/annotations/sample1.txt --maximum_allowed_distance 50 --maximum_limbo_lifetime 60 
  ```

5) Specify the crossing in the cropped image then run `classify_trajectories.py` to determine pedestrians that crossed the road. Running `classify_trajectories.py` appends the columns `cross` that states if the pedestrian eventually crossed the road, and `incrossing` that states if the pedestrian is currently in the crossing or is at the sidewalks. Note that the value of `cross` for each pedestrian will be similar throughout his lifetime.

  ```bash
  cd ROOT 
  python3 ./annotator/classify_trajectories.py --filename ../dataset/all/annotations/sample1-modified.txt --tl 565 70 --tr 650 70 --br 650 300 --bl 465 300
  ```

![Alt Text](/images/dataset/step2.png)

6) Crop the pedestrians to create the dataset. The images of each pedestrian will be located at `/dataset/all/crops/<video_name>/<pedestrian_id>/<frame_number>.png`. For example, the image of pedestrian 2 at frame 1000 for the video `sample1.MP4` will be located at `/dataset/all/crops/sample1/0000000002/0000001000.png`

7) OPTIONAL: The results of the tracker can be visualized by running `annotate_video.py`

We remind the users that a shell script including all the steps have been included at `/annotator/annotate-sample.sh` and that the folders `/dataset/all/crops/sample1` and `/dataset/all/annotations/sample1.txt` must be moved to `/dataset/train/crops/sample1` and `/dataset/train/annotations/sample1.txt` for it to be used in the training set. 

# Training

We built a simple CNN LSTM as a baseline for our study. Details are in the report. Run `train.sh` to train the architecture. The model will be saved at `/pedestrian_intention/models/`. 

# Results
------------
## Prediction

Run `evaluate_lausanne.ipynb` to get visual results when classifying at every timestep. In the gif below, a green bounding box indicates a decision of "not crossing" while a red bounding box indicates a decision of "crossing".

![Alt Text](/images/prediction/Ouchy1.gif) ![Alt Text](/images/prediction/Ouchy2.gif) 
![Alt Text](/images/prediction/Riponne1.gif) ![Alt Text](/images/prediction/Riponne2.gif) 

## Guided backpropagation

Run `guidedbackprop_lausanne.ipynb` to get visual results when classifying at every timestep. In the images below, a green bounding box indicates a decision of "not crossing" while a red bounding box indicates a decision of "crossing".

![Alt Text](/images/guidedbackprop/crossing1.png) 
![Alt Text](/images/guidedbackprop/crossing2.png)
![Alt Text](/images/guidedbackprop/notcrossing1.png)
