import os
import cv2
import sys
import glob
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm

# ------- args ------- # 
in_filepath = sys.argv[1] 
im_folderpath = sys.argv[2]
out_crop_folderpath = sys.argv[3]
min_lifetime_noncrossers = 240
min_lifetime_crossers = 60
# ------- args ------- #

# ------- utils ------- #
def textfile_to_array(filename, dtype=float):    
    with open(filename) as file:
        data = file.readlines()
        data = [list(map(float, x.split(","))) for x in data]
        data = np.array(data, dtype=dtype)        
    return data
# ------- utils ------- #

# ------- main ------- #
df = pd.read_csv(in_filepath)
#df = df.astype('int')

# prepare folders
# delete folder if it exists
if os.path.exists(out_crop_folderpath) and os.path.isdir(out_crop_folderpath):
    shutil.rmtree(out_crop_folderpath, ignore_errors=True)
# create all folders
if not os.path.exists(out_crop_folderpath):
    os.makedirs(os.path.join(out_crop_folderpath))
for i in df[(df["cross_true"] == 1) & (df["lifetime"] > min_lifetime_crossers)]["id"].unique():
    directory = os.path.join(out_crop_folderpath, str(i).zfill(10))
    if not os.path.exists(directory):
        os.makedirs(directory)
for i in df[(df["cross_true"] == 0) & (df["lifetime"] > min_lifetime_noncrossers)]["id"].unique():
    directory = os.path.join(out_crop_folderpath, str(i).zfill(10))
    if not os.path.exists(directory):
        os.makedirs(directory)
		
cap = cv2.VideoCapture(im_folderpath)
video_frame_counter = 0
   
new_rows = []
print("Cropping pedestrians")
pbar = tqdm(total = df["frame"].nunique())
for tt in df["frame"].unique():    
    t = tt - 1
    
    ret=True
    while(video_frame_counter <= t):
        # get frame
        ret, im = cap.read()
        if(ret==False):
            break
        video_frame_counter+=1
    if(ret==False):
        break
    
    pbar.update(1)
    
    for row in df[df["frame"] == t].itertuples(index=False, name='Pandas'):
        
        if((row.height <= 50) or (row.tly + row.height + 40 > np.shape(im)[0])):
            continue
						
        tly = np.maximum(0,int(row.tly - 0.10*np.float(row.height)))
        tlx = np.maximum(0,int(row.tlx - 0.10*np.float(row.width)))
        brx = np.minimum(np.shape(im)[1],int(row.tlx + row.width + 0.10*np.float(row.width)))
						
        if(row.cross_true == 1 and row.lifetime > min_lifetime_crossers): # and row.incrossing == 0
            #crop = im[tly:row.tly+row.height, row.tlx:brx, :]
            #cv2.imwrite(os.path.join(out_crop_folderpath, str(row.id).zfill(10), str(t).zfill(10) + ".png"), crop)
            new_rows.append(row)
            
        if(row.cross_true == 0 and row.lifetime > min_lifetime_noncrossers): #  and row.incrossing == 0 
            #crop = im[tly:row.tly+row.height, row.tlx:brx, :]
            #cv2.imwrite(os.path.join(out_crop_folderpath, str(row.id).zfill(10), str(t).zfill(10) + ".png"), crop)
            new_rows.append(row)
            
        #print(row)
        #print(df.columns)
        #input()
            
df_final = pd.DataFrame(new_rows, columns=df.columns)
df_final.to_csv("test.txt", index=False)