import os
import sys
import cv2
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

# ------- args ------- # 
min_lifetime_noncrossers = 240
min_lifetime_crossers = 120
in_video    = sys.argv[1] # "./videos/23-10-2018/Ouchy-2-Left-cropped.MP4" 
op_video    = sys.argv[2] # "./videos/23-10-2018/Ouchy-2-Left-cropped-drawn.MP4"
filename = sys.argv[3] # "./annotations/Ouchy-2-Left"
# ------- args ------- #

# ------- utils ------- #
def get_bbox_position(arr, dtype=float):
    return dtype(arr[0]+arr[2]/2), dtype(arr[1]+arr[3])
# ------- utils ------- #

# ------- main ------- #
df = pd.read_csv(filename)
df = df.astype(int)
df = df[df["height"] > 50]

in_cap = cv2.VideoCapture(in_video)
op_cap = cv2.VideoWriter(op_video, cv2.VideoWriter_fourcc('H','2','6','4'), 30, (int(in_cap.get(3)),int(in_cap.get(4))))
       
print("Drawing trajectories")
pbar = tqdm(total = df["frame"].max())
t = 1
while(in_cap.isOpened()):
    
    pbar.update(1)

    # get frame
    ret, im = in_cap.read()
    if(ret==False):
        break

    for row in df[df["frame"] == t].itertuples(index=True, name='Pandas'):
	
        #if((row.height <= 50) or (row.tly + row.height + 40 > np.shape(im)[0])):
	
        # red box for crossing
        if(row.cross == 1 and row.lifetime > min_lifetime_crossers):
            cv2.putText(im , "crossing", (row.tlx, row.tly+row.height+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(im , str(row.id), (row.tlx, row.tly+row.height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(im, (row.tlx, row.tly), (row.tlx+row.width, row.tly+row.height), color=(0, 0, 255), thickness=2)
                   
        if(row.cross == 0 and row.lifetime > min_lifetime_noncrossers):
            cv2.putText(im , str(row.id), (row.tlx, row.tly+row.height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(im, (row.tlx, row.tly), (row.tlx+row.width, row.tly+row.height), color=(255,0,0), thickness=2)

    op_cap.write(im)
    t+=1
in_cap.release()
op_cap.release() 
# ------- main ------- #
