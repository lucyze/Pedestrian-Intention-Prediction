#!/bin/bash

###############################
#                             #
#                             #
# REMEMBER TO RUN ON GIT BASH # 
# IF ON WINDOWS               #
#                             #
###############################

# reassign to proper variable names
# so its easier to read and debug

filename=$1

crop_video=$2
run_detector=$3
generate_groundtruth=$4
annotate_video=$5
crop_pedestrian=$6

echo crop_video ${crop_video}
echo run_detector ${run_detector}
echo generate_groundtruth ${generate_groundtruth}
echo annotate_video ${annotate_video}
echo crop_pedestrians ${crop_pedestrian}

if [ "$crop_video" -eq 1 ]
then
	echo Cropping video
	ffmpeg -y -i ../dataset/all/raw/${filename}.MP4 -filter:v "crop=${9}:${10}:${7}:${8}" ../dataset/all/processed/${filename}.MP4
fi

if [ "$run_detector" -eq 1 ]
then
	echo Running detector	
	cd ./mask-rcnn.pytorch
	python3 -u tools/infer_simple.py --dataset keypoints_coco2017 --cfg ./configs/baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml --load_detectron ./data/weights/R-50-FPN-1x.pkl --image_dir ../../dataset/all/processed/${filename}.MP4 --output_dir ../../dataset/all/annotations/${filename} # next time add --annotations and --video
  cd ..
fi
	
if [ "$generate_groundtruth" -eq 1 ]
then
	echo Assigning labels and classifying trajectories
	python3 ./scripts/hungarian.py ../dataset/all/annotations/${filename}.txt ${15} ${16}
	python3 ./scripts/classify_trajectories.py --filename ../dataset/all/annotations/${filename}-modified.txt --tl ${11} ${12} --tr ${13} ${14} --br ${15} ${16} --bl ${17} ${18}
fi

if [ "$annotate_video" -eq 1 ]
then
	echo Drawing trajectories video
	python3 ./scripts/draw_trajectories.py ../dataset/all/processed/${filename}.MP4 ../dataset/all/processed/${filename}-annotated.MP4 ../dataset/all/annotations/${filename}-modified.txt
fi

if [ "$crop_pedestrian" -eq 1 ]
then
	echo Cropping bounding boxes to ~/datasets/filename/crops
	rm -r ../dataset/all/crops/${filename}-crops
	mkdir ../dataset/all/crops/${filename}-crops
	python ./scripts/crop_pedestrians.py ../dataset/all/annotations/${filename}-modified.txt ../dataset/all/processed/${filename}.MP4 ../dataset/all/crops/${filename}-crops
fi

echo Done!
cmd /k
