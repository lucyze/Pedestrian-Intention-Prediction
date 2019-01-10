import pandas as pd
import numpy as np
import glob
import sys
import os

from tqdm import tqdm
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# ------- args ------- #
filename = sys.argv[1] # "./annotations/Ouchy-1-Right-Short.txt"
max_distance = int(sys.argv[2])
max_lifetime = int(sys.argv[3])
# ------- args ------- #

# ------- utils ------- #
def get_bbox_position(arr, dtype=float):
    return dtype(arr[0]+arr[2]/2), dtype(arr[1]+arr[3])
# ------- utils ------- #

# ------- main ------- #
df = pd.read_csv(filename) 
df = df[df["score"] > 0.8]

# initialize label for time t=0
print("\nRunning Hungarian tracker")
pbar = tqdm(total = int(df["frame"].max()))
Y = [[i for i in range(len(df[df["frame"] == 0]))]]
buf = []
for t in range(int(df["frame"].max())):
                   
    pbar.update(1)
        
	# get the pedestrian height at time t and t+1
    h1 = np.asarray(df[df["frame"] == t][["height"]].values).reshape(1, -1)[0,:]
    h2 = np.asarray(df[df["frame"] == t+1][["height"]].values).reshape(1, -1)[0,:]
		
    # get the pedestrian bounding boxes at time t and t+1
    # x1 is t
    # x2 is t+1
    x1 = df[df["frame"] == t][["tlx","tly","width","height"]].values
    x2 = df[df["frame"] == t+1][["tlx","tly","width","height"]].values
    x1 = np.array([get_bbox_position(bbox, dtype=float) for bbox in x1])
    x2 = np.array([get_bbox_position(bbox, dtype=float) for bbox in x2]) 
        
    # append Y for time t+1
    Y.append([-1 for i in range(np.shape(x2)[0])])
                       
    # append items in buffer to x1 (coordinates) and h1 (height)
    x1_buf = list(x1)
    h1_buf = list(h1)
    for b in buf:
        x1_buf.append(b[2])
        h1_buf.append(b[3])
    x1_buf = np.array(x1_buf)
    h1_buf = np.array(h1_buf)
	
	#h1_buf = list(h1)
    #for b in buf_h:
    #    h1_buf.append(b)
    #h1_buf = np.array(h1_buf)
        
    # - compute the distances between all pedestrians at time t (and in the buffer) to time t+1
    # - then compute the optimal assignment
    # *** only compute assignments if x2 is not empty
    row_index = []
    col_index = []
    if(np.shape(x2)[0] != 0):      
        
        # compute the optimal assignments between x1 and x2 first
        row_index_priority = []
        col_index_priority = []
        row_assigned = []
        col_assigned = []
        if(np.shape(x1)[0] != 0):
            distances_priority = distance.cdist(x1, x2, 'euclidean')
            row_index_priority, col_index_priority = linear_sum_assignment(distances_priority)
            for r,c in zip(row_index_priority, col_index_priority):
                # assign labels from time t to t+1
                if((distances_priority[r][c] < max_distance) and (np.float(np.minimum(h1[r],h2[c]))/np.float(np.maximum(h1[r],h2[c])) > 0.5)):
                    Y[-1][c] = Y[-2][r]  
                    row_assigned.append(r)
                    col_assigned.append(c)
                
        # now compute the optimal assignments between x1_buf and x2 without double assignments
        # set the distances between the previously matched objects to a very small number to force them to match again
        if(np.shape(x1_buf)[0] != 0):
            distances = distance.cdist(x1_buf, x2, 'euclidean')        
            for r,c in zip(row_assigned, col_assigned):
                distances[r][c] = -999999             
            row_index, col_index = linear_sum_assignment(distances)
            rm = []
            for r,c in zip(row_index, col_index):
                # assign labels from time t to t+1
                #if(r < np.shape(x1)[0]):
                #    if(distances[r][c] < 50):
                #        Y[-1][c] = Y[-2][r]  
                # assign labels from buffer to t+1
                # then remove them
                if(r >= np.shape(x1)[0]):
                    if((distances[r][c] < max_distance) and  (np.float(np.minimum(h1_buf[r],h2[c]))/np.float(np.maximum(h1_buf[r],h2[c])) > 0.5)):
                        buf_ind = r - np.shape(x1)[0]
                        rm.append(buf_ind)
                        Y[-1][c] = Y[buf[buf_ind][0]][buf[buf_ind][1]]
            # remove items from buffer that were assigned
            buf_temp = []
            for i,b in enumerate(buf):
                if(i not in rm):
                    buf_temp.append(b)
            buf = buf_temp.copy()
                
    # *** only compute assignments if x2 is not empty
            
    # iterate through x1 and check if it has been assigned to an object in x2
    # if not then add it to the buffer
    for i in range(np.shape(x1)[0]):
        if(i not in row_index):
            tup = (-2, i, x1[i], h1[i]) # (timestamp, index of object in x1, coordinates)
            buf.append(tup)
               
    # give new label to unassigned objects in x2
    # these are (most probably) newly found objects
    for i in range(len(Y[-1])):
        if(Y[-1][i] == -1):
            Y[-1][i] = np.amax([y for yy in Y for y in yy]) + 1
            
    # increment timestamps
    for i in range(len(buf)):
        buf[i] = (buf[i][0] - 1, buf[i][1], buf[i][2], buf[i][3])
        
    # remove items from buffer that has exceeded the maximum lifetime
    rm = []
    for i,b in enumerate(buf):
        if(b[0] < -1*max_lifetime):
            rm.append(i)
    buf_temp = []
    for i,b in enumerate(buf):
        if(i not in rm):
            buf_temp.append(b)
    buf = buf_temp.copy()

# rewrite ground truth with label
print("\nAssigning labels")
pbar = tqdm(total = len(Y))
for t,y in enumerate(Y):    
    pbar.update(1)
    df.loc[df["frame"] == t,"id"] = y
df.to_csv(filename[0:-4]+"-modified.txt", index=False)
#np.savetxt(filename[0:-4]+"-modified.txt", df.values, fmt='%f', delimiter=",")
print("Hungarian complete !")