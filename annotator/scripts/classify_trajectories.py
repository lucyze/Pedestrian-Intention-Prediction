import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

print("Classifying trajectories")

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--tl', nargs='+', type=int)
parser.add_argument('--tr', nargs='+', type=int)
parser.add_argument('--br', nargs='+', type=int)
parser.add_argument('--bl', nargs='+', type=int)

args = parser.parse_args()
tl = tuple(args.tl)
tr = tuple(args.tr)
br = tuple(args.br)
bl = tuple(args.bl)

#print(tl,tr,br,bl, args.folder)

# ------- args ------- # 
filename = args.filename #"./ground-truth/Ouchy-1/*.txt"
crossing = np.array([tl, tr, br, bl, tl], dtype=np.int)
# ------- args ------- # 

# ------- utils ------- #
def get_bbox_position(arr, dtype=float):
    return dtype(arr[0]+arr[2]/2), dtype(arr[1]+arr[3])
# ------- utils ------- #

# ------- main ------- #
df = pd.read_csv(filename) 
df["lifetime"] = 0
df["cross"] = 0
df["incrossing"] = 0
#df = textfile_to_array(in_filepath, float)
#df = sorted(df, key=lambda x : x[0])
#df = pd.DataFrame(df, columns=["frame", 'id', 'tlx', 'tly', 'width', 'height', 'score', 'cross', 'incrossing', 'x'])
            
print("\nChecking whether each pedestrian crossed the street and computing his lifetime")
pbar = tqdm(total = df["id"].nunique())
for i in df["id"].unique():
    pbar.update(1)
    
    # get the trajectory for the current id
    bboxes = df[df["id"] == i][["tlx", "tly", "width", "height"]].values
    trajectories = np.array([get_bbox_position(bbox, dtype=float) for bbox in bboxes])
        
    # check if pedestrian crossed the road
    cross = 0
    incrossing = [0 for i in range(len(trajectories))]

	# 1. check if the last 20% of his trajectories is in the crossing
    for t in range(int(np.ceil(0.8*len(trajectories))), len(trajectories)):
        in_crossing = cv2.pointPolygonTest(crossing,(trajectories[t][0],trajectories[t][1]), False)
        if(in_crossing == 1):
            cross = 1  
            incrossing[t] = 1
			
	# 2. check if it started in the crossing	
    df.loc[df["id"] == i, "cross"] = cross  
    df.loc[df["id"] == i, "incrossing"] = incrossing
    
    # compute his lifetime
    if(cross == 0):
        df.loc[df["id"] == i, "lifetime"] = df[df["id"] == i]["frame"].max() - df[df["id"] == i]["frame"].min()
    if(cross == 1):
        df.loc[df["id"] == i, "lifetime"] = df[(df["id"] == i) & (df["incrossing"] == 0)]["frame"].max() - df[(df["id"] == i) & (df["incrossing"] == 0)]["frame"].min()     
df.to_csv(filename[0:-4]+".txt", index=False)
print("Done assigning labels to pedestrians")