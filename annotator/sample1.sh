#!/bin/bash

# file arguments
# 1) filename=Ouchy-1

# processing options
# 2) crop_video 
# 3) run_detector
# 4) generate_groundtruth
# 5) annotate_video
# 6) crop_pedestrian

# crop coordinates
# 7) top left x
# 8) top left y
# 9) width
# 10) height

# road or zebra crossing coordinates
# 11) top left point
# 12) top right point
# 13) bottom right point
# 14) bottom left point

# tracker parameters
# 15) max distance
# 16) max lifetime

# KEEP
./run-annotator.sh \
sample2 \
0 0 1 0 1 \
0 875 650 250 \
565 70 \
650 70 \
650 300 \
465 300 \
50 60 

## KEEP
#./run-pipeline.sh \
#Lausanne-Gare-2-Right 23-10-2018 e130167 35.203.145.83 id_rsantu \
#0 0 0 1 1 0 \
#950 630 175 450 \
#0 110 \
#50 110 \
#145 145 \
#0 145 \
#0

# KEEP
#./run-pipeline.sh \
#Ouchy-2-Right 23-10-2018 e130167 104.196.250.78 id_rsantu \
#0 0 1 1 1 0 \
#1000 945 200 500 \
#0 100 \
#50 100 \
#130 130 \
#0 130 \
#0

#./run-pipeline.sh \
#Ouchy-1-Right-Short 23-10-2018 HaziqBinRazali03 35.233.220.36 id_rsa03 \
#0 1 0 0 0 0 \
#1150 600 120 770 \
#0 85 \
#40 85 \
#150 110 \
#0 210 \
#0

#./run-pipeline.sh \
#Ouchy-1 23-10-2018 HaziqBinRazali01 35.233.144.214 id_rsa \
#0 0 0 0 0 1 \
#0 600 180 1920 \
#670 95 \
#1100 85 \
#1300 110 \
#550 130 \

#./run-pipeline.sh \
#Ouchy-2 23-10-2018 HaziqBinRazali01 35.233.144.214 id_rsa \
#0 0 1 1 1 1 \
#0 900 540 1920 \
#550 145 \
#1050 140 \
#1150 170 \
#530 190 \
