# Pedestrian Intention Prediction

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Dataset](#usage)
  * [Results](#results)
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
        ├── raw                        : raw recordings  
        ├── input                      : recordings set as input to the mask rcnn
        ├── annotations                : mask rcnn output. the (approximate) ground truth
    
    ├── annotator                      
        ├── mask-rcnn.pytorch          : mask rcnn people detector 
        ├── tracker                    : hungarian tracker 
        ├── annotate.sh                : shell script that runs the full annotation pipeline
     
    ├── intention_prediction
        ├── models                     : pretrained models 
        ├── results                    : 
        ├── scripts                    : scripts containing the models and data loader
        ├── train.py                   : train script
        ├── train.sh                   : train script
        ├── test.ipynb                 : test ipython script (guided backprop, evaluation)
        
    ├── README.md                      : the README guideline and explanation for our project.
    ├── report.pdf                     : report
    ├── slides_midterm.pptx            : mid-term presentation slides
    ├── slides_final.pptx              : final presentation slides

# Dataset
------------

To augment the dataset of [1], we have that can be automatically annotated. In the following, we describe the steps needed to generate (approximate) the ground truths of your own recordings. A shell script including all the steps have been included.

1)	Decide on the region of interest that the annotator will operate on in the following order (top left x coordinate, top left y coordinate, bottom right x coordinate, bottom right y coordinate). The coordinates in this example are (1,2,3,4). Crop this region via ffmpeg and place the output in .\dataset\input\filename

2)	Run the Mask RCNN people detector on the input video at \dataset\input\filename. The output is a csv containing the detections.

| Frame no  | UID | tlx | tlx | width | height | score |  
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | 0 | 474 | 12 | 20 | 56 | 0.995529 |
| 1 | 0 | 474 | 12 | 20 | 56 | 0.995529 |

3)	Specify the crossing in the cropped image.

4)	Convert the video 

#### Testing
Launch `models\moving-mnist\sample-test.ipynb` 

#### Training
The training dataset was too large to be uploaded to github (~2GB). To train, run `datasets\gen-moving-mnist.ipynb` with the default main arguments to generate the dataset. Then run `models\moving-mnist\sample-train.ipynb`. Note that the saved model `model.ckpt` will be overwritten.

# Results
------------
## Long-Term Prediction

Reconstruction results for 2 input objects. The sequence is 200 frames long and both models received the ground truth as input for the first 20 frames. **Top: Ground Truth, Middle: RNN, Bottom: VRNN**

![Alt Text](/results/moving-shapes/2/0-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/1-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/2-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/3-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/4-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/5-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/6-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/7-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/8-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/9-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/10-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/11-2-shapes.gif) ![Alt Text](/results/moving-shapes/2/12-2-shapes.gif)

![Alt Text](/results/moving-mnist/2/0-2-digits.gif) ![Alt Text](/results/moving-mnist/2/1-2-digits.gif) ![Alt Text](/results/moving-mnist/2/2-2-digits.gif) ![Alt Text](/results/moving-mnist/2/3-2-digits.gif) ![Alt Text](/results/moving-mnist/2/4-2-digits.gif) ![Alt Text](/results/moving-mnist/2/5-2-digits.gif) ![Alt Text](/results/moving-mnist/2/6-2-digits.gif) ![Alt Text](/results/moving-mnist/2/7-2-digits.gif) ![Alt Text](/results/moving-mnist/2/8-2-digits.gif) ![Alt Text](/results/moving-mnist/2/9-2-digits.gif) ![Alt Text](/results/moving-mnist/2/10-2-digits.gif) ![Alt Text](/results/moving-mnist/2/11-2-digits.gif) ![Alt Text](/results/moving-mnist/2/12-2-digits.gif)

## Out of Domain Runs (1 input object)

Out of domain runs with 3 input objects. The sequence is 40 frames long and both models received the ground truth as input for the first 20 frames. **Top: Ground Truth, Middle: RNN, Bottom: VRNN**

![Alt Text](/results/moving-shapes/1/0-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/1-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/2-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/3-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/4-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/5-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/6-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/7-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/8-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/9-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/10-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/11-1-shapes.gif) ![Alt Text](/results/moving-shapes/1/12-1-shapes.gif)

![Alt Text](/results/moving-mnist/1/0-1-digits.gif) ![Alt Text](/results/moving-mnist/1/1-1-digits.gif) ![Alt Text](/results/moving-mnist/1/2-1-digits.gif) ![Alt Text](/results/moving-mnist/1/3-1-digits.gif) ![Alt Text](/results/moving-mnist/1/4-1-digits.gif) ![Alt Text](/results/moving-mnist/1/5-1-digits.gif) ![Alt Text](/results/moving-mnist/1/6-1-digits.gif) ![Alt Text](/results/moving-mnist/1/7-1-digits.gif) ![Alt Text](/results/moving-mnist/1/8-1-digits.gif) ![Alt Text](/results/moving-mnist/1/9-1-digits.gif) ![Alt Text](/results/moving-mnist/1/10-1-digits.gif) ![Alt Text](/results/moving-mnist/1/11-1-digits.gif) ![Alt Text](/results/moving-mnist/1/12-1-digits.gif)

## Out of Domain Runs (3 input objects)

Out of domain runs with 3 input objects. The sequence is 40 frames long and both models received the ground truth as input for the first 20 frames. **Top: Ground Truth, Middle: RNN, Bottom: VRNN**

![Alt Text](/results/moving-shapes/3/0-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/1-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/2-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/3-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/4-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/5-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/6-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/7-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/8-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/9-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/10-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/11-3-shapes.gif) ![Alt Text](/results/moving-shapes/3/12-3-shapes.gif)

![Alt Text](/results/moving-mnist/3/0-3-digits.gif) ![Alt Text](/results/moving-mnist/3/1-3-digits.gif) ![Alt Text](/results/moving-mnist/3/2-3-digits.gif) ![Alt Text](/results/moving-mnist/3/3-3-digits.gif) ![Alt Text](/results/moving-mnist/3/4-3-digits.gif) ![Alt Text](/results/moving-mnist/3/5-3-digits.gif) ![Alt Text](/results/moving-mnist/3/6-3-digits.gif) ![Alt Text](/results/moving-mnist/3/7-3-digits.gif) ![Alt Text](/results/moving-mnist/3/8-3-digits.gif) ![Alt Text](/results/moving-mnist/3/9-3-digits.gif) ![Alt Text](/results/moving-mnist/3/10-3-digits.gif) ![Alt Text](/results/moving-mnist/3/11-3-digits.gif) ![Alt Text](/results/moving-mnist/3/12-3-digits.gif) 

## Generation diversity

We test the VRNN's capability in generating novel sequences on 4 separate runs .  Each sequence is 200 frames long and both models received the ground truth as input for the first 20 frames. **Top: Ground Truth. Each row is a separate run.**

![Alt Text](/results/moving-shapes/diversity/0-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/1-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/2-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/3-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/4-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/5-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/6-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/7-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/8-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/9-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/10-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/11-2-shapes.gif) ![Alt Text](/results/moving-shapes/diversity/12-2-shapes.gif)

![Alt Text](/results/moving-mnist/diversity/6-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/7-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/8-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/9-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/10-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/11-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/12-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/13-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/14-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/15-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/16-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/17-2-digits.gif) ![Alt Text](/results/moving-mnist/diversity/18-2-digits.gif)
