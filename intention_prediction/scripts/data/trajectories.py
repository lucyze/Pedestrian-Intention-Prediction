import os
import cv2
import math
import glob
import random
import logging

import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

def seq_collate(data):
    
    # calls the __getitem__ method
    (pedestrian_images, pedestrian_labels, pedestrian_filenames) = zip(*data)

    pedestrian_labels = np.array(pedestrian_labels)
    pedestrian_labels = torch.from_numpy(pedestrian_labels).type(torch.long)
    return [pedestrian_images, pedestrian_labels, pedestrian_filenames]

def Lausannecollate(data):
    
    # calls the __getitem__ method
    
    (pedestrian_images, pedestrian_labels, pedestrian_foldernames, pedestrian_filenames) = zip(*data)

    pedestrian_labels = np.array(pedestrian_labels)
    pedestrian_labels = torch.from_numpy(pedestrian_labels).type(torch.long)
    return [pedestrian_images, pedestrian_labels, pedestrian_foldernames, pedestrian_filenames]

def JAADcollate(data):
        
    # calls the __getitem__ method
    (pedestrian_images, standing, looking, walking, crossing, pedestrian_foldernames, pedestrian_filenames) = zip(*data)

    # cannot convert to tensor due to variable length
    #standing = torch.tensor(standing)  
    #looking = torch.tensor(looking)    
    #walking = torch.tensor(walking)
    #crossing = torch.tensor(crossing)
    
    return [pedestrian_images, standing, looking, walking, crossing, pedestrian_foldernames, pedestrian_filenames]

def read_file(_path, delim="\t"):
    data = []
    if delim == 'tab':
        delim = "\t"
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split("	") #line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

# ------- utils ------- #
def textfile_to_array(filename, dtype=float):    
    with open(filename) as file:
        data = file.readlines()
        data = [list(map(float, x.split())) for x in data]
        #data = np.array(data, dtype=dtype)        
    return data

def LausanneDataset(data_dir, min_obs_len=8, max_obs_len=8, timestep=10):
    
        debug_op = open("debug_op.txt", "w")
        # read annotations  
        df = pd.DataFrame()
        print("Reading annotations from ", data_dir)
        for file in glob.glob(os.path.join(data_dir,"annotations","*")):
            print("Reading ", file)
            df = df.append(pd.read_csv(file), ignore_index=True) 
            
        # assign unique id to each pedestrian
        df["unique_id"] = df.groupby(['folderpath']).ngroup()     
                               
        # --------------------------------------------------------------
        # get pedestrian frames in steps of 'timestep'
        # lousier than newer implementation since it takes the min, potentially missing out the final crossing decision
        ind = []
        print(" Processing sequence")
        pbar = tqdm(total = int(df["unique_id"].nunique()))
        for i in df["unique_id"].unique():
            pbar.update(1)
            ind_temp = []
            # get dataframe of pedestrian i
            df_temp = df[df["unique_id"] == i]     
            curr_frame = df_temp["frame"].min()-1
            # collect frame at every timestep
            while curr_frame <= df_temp["frame"].max():
                try:
                    ind_temp.append(np.min(df_temp[(df_temp["frame"] >= curr_frame) & (df_temp["frame"] < curr_frame + timestep)].index.tolist()))
                except ValueError:
                    pass
                curr_frame+=timestep
            # retain sequence if its length is satisfied
            if(len(ind_temp) >= min_obs_len):
                ind = ind + ind_temp
        df = df.iloc[ind].reset_index(drop=True)
        df["unique_id"] = df.groupby(['folderpath']).ngroup() 
                
        ## --------------------------------------------------------------
        ## get pedestrian frames in steps of 'timestep'
        ## better than older implementation since it takes the max,making it impossible to miss out the final crossing decision
        ## need to sort the final indices
        #ind = []
        #print(" Processing sequence")
        #pbar = tqdm(total = int(df["unique_id"].nunique()))
        #for i in df["unique_id"].unique():
        #    pbar.update(1)
        #    ind_temp = []
        #    # get dataframe of pedestrian i
        #    df_temp = df[df["unique_id"] == i]     
        #    curr_frame = df_temp["frame"].max()
        #    # collect frame at every timestep
        #    while curr_frame >= df_temp["frame"].min():
        #        try:
        #            ind_temp.append(np.max(df_temp[(df_temp["frame"] <= curr_frame) & (df_temp["frame"] > curr_frame - timestep)].index.tolist()))
        #        except ValueError:
        #            pass
        #        curr_frame-=timestep
        #    # retain sequence if its length is satisfied
        #    if(len(ind_temp) >= min_obs_len):
        #        ind = ind + ind_temp
        #        debug_op.write("Keeping "+str(df_temp["folderpath"].iloc[-1])+" that has been truncated from "+str(df_temp["frame"].max()-df_temp["frame"].min()+1)+" frames to "+str(len(ind_temp))+" frames\n")
        #    else:
        #        debug_op.write("Removing "+str(df_temp["folderpath"].iloc[-1])+" that has been truncated from "+str(df_temp["frame"].max()-df_temp["frame"].min()+1)+" frames to "+str(len(ind_temp))+" frames\n")
        #ind = sorted(ind)
        #df = df.iloc[ind].reset_index(drop=True)
        #df["unique_id"] = df.groupby(['folderpath']).ngroup()
               
        # finalize dataframe 
        df = df.groupby("unique_id").agg(lambda x: list(x)) 
        
        crosser_true = 0
        crosser_false = 0
        # iterate through dataframe
        # count number of one and zero
        for row in df.itertuples(index=False, name='Pandas'):
            if(row.cross_true[0] == 0):
                crosser_true += 1
            if(row.cross_true[0] == 1):
                crosser_false += 1
                
        print(crosser_true, crosser_false)
        input()
        
        # optional. for debugging purposes
        # view total number of crossing / non-crossing scenarios
        #debug_op.write("\n")
        #crossers = 0
        #noncrossers = 0
        #for i in range(len(df)):
        #    df_temp = df.iloc[i]
        #    if(df_temp["crossing"][-1] == 1):
        #        crossers+=1
        #        debug_op.write("crosser "+str(df_temp["folderpath"][-1])+"\n")
        #        debug_op.write("standing "+str(df_temp["standing"][-1*max_obs_len:])+"\n")
        #        debug_op.write("looking "+str(df_temp["looking"][-1*max_obs_len:])+"\n")
        #        debug_op.write("walking "+str(df_temp["walking"][-1*max_obs_len:])+"\n")
        #        debug_op.write("crossing "+str(df_temp["crossing"][-1*max_obs_len:])+"\n")
        #        debug_op.write("\n")
        #    else:
        #        noncrossers+=1
        #        debug_op.write("non-crosser "+str(df_temp["folderpath"][-1])+"\n")
        #        debug_op.write("standing "+str(df_temp["standing"][-1*max_obs_len:])+"\n")
        #        debug_op.write("looking "+str(df_temp["looking"][-1*max_obs_len:])+"\n")
        #        debug_op.write("walking "+str(df_temp["walking"][-1*max_obs_len:])+"\n")
        #        debug_op.write("crossing "+str(df_temp["crossing"][-1*max_obs_len:])+"\n")
        #        debug_op.write("\n")
        
        #debug_op.write("no crossers: "+str(crossers)+"\n")
        #debug_op.write("no noncrossers: "+str(noncrossers)+"\n")
        #debug_op.close()
        #print("no crossers: ", crossers)
        #print("no noncrossers: ", noncrossers)   
        #display(df.iloc[0:40])
        
        return df

#def LausanneDataset(data_dir, min_obs_len=8, max_obs_len=8, timestep=10):
    
#        debug_op = open("debug_op.txt", "w")
#        # read annotations  
#        df = pd.DataFrame()
#        print("Reading annotations from ", data_dir)
#        for file in glob.glob(os.path.join(data_dir,"annotations","*")):
#            print("Reading ", file)
#            df = df.append(pd.read_csv(file), ignore_index=True) 
#            
#        # keep only those that interacted with the driver
#        df = df[df["type"] == "pedestrian"].reset_index(drop=True) 
#         
#        # assign unique id to each pedestrian
#        df["unique_id"] = df.groupby(['folderpath']).ngroup()     
#        
#        # ADDING
#        # keep only those that are not under occlusion
#        df = df[df["occlusion"] == 0].reset_index(drop=True)      
#            
#        # --------------------------------------------------------------
#        # retain sequence for each pedestrian up till he begins to cross 
#        ind = []
#        print(" Processing sequence")
#        pbar = tqdm(total = int(df["unique_id"].nunique()))
#        for i in df["unique_id"].unique():
#            pbar.update(1)
#            ind_temp = []
#            df_temp = df[df["unique_id"] == i]  
#            #for folderpath, filename in zip(df_temp["folderpath"][-1*min_obs_len:], df_temp["filename"][-1*min_obs_len:]):
#            #    print(os.path.join(folderpath,filename))
#            #print(len(np.where(df_temp['crossing']>0)[0]))
#            #print(len(df_temp[df_temp['crossing']==1]))
#            #print(df_temp['crossing'])
#            # pedestrian interacts with driver but does not cross  
#            if((len(np.where(df_temp['crossing_true']>0)[0])==0) and (len(df_temp[df_temp['crossing_true']==1])==0)):
#                ind_temp = list(df_temp.index.values)
#                ind = ind + ind_temp
#                #print(ind_temp)
#                continue
#                                            
#            # retain the indices up till pedestrian begins to cross
#            ind_temp = list(df_temp.iloc[0:np.min(np.where(df_temp['crossing_true']>0))+1].index.values)
#            ind = ind + ind_temp
#            #if(len(ind_temp) >= min_obs_len): # remove when uncommenting below
#            #    ind = ind + ind_temp
#        df = df.iloc[ind].reset_index(drop=True)
#        df["unique_id"] = df.groupby(['folderpath']).ngroup() 
#        # ADDING
#                       
#        ## --------------------------------------------------------------
#        ## get pedestrian frames in steps of 'timestep'
#        ## lousier than newer implementation since it takes the min, potentially missing out the final crossing decision
#        #ind = []
#        #print(" Processing sequence")
#        #pbar = tqdm(total = int(df["unique_id"].nunique()))
#        #for i in df["unique_id"].unique():
#        #    pbar.update(1)
#        #    ind_temp = []
#        #    # get dataframe of pedestrian i
#        #    df_temp = df[df["unique_id"] == i]     
#        #    curr_frame = df_temp["frame"].min()-1
#        #    # collect frame at every timestep
#        #    while curr_frame <= df_temp["frame"].max():
#        #        try:
#        #            ind_temp.append(np.min(df_temp[(df_temp["frame"] >= curr_frame) & (df_temp["frame"] < curr_frame + #timestep)].index.tolist()))
#        #        except ValueError:
#        #            pass
#        #        curr_frame+=timestep
#        #    # retain sequence if its length is satisfied
#        #    if(len(ind_temp) >= min_obs_len):
#        #        ind = ind + ind_temp
#        #df = df.iloc[ind].reset_index(drop=True)
#        #df["unique_id"] = df.groupby(['folderpath']).ngroup() 
#                       
#        # --------------------------------------------------------------
#        # get pedestrian frames in steps of 'timestep'
#        # better than older implementation since it takes the max,making it impossible to miss out the final crossing decision
#        # need to sort the final indices
#        ind = []
#        print(" Processing sequence")
#        pbar = tqdm(total = int(df["unique_id"].nunique()))
#        for i in df["unique_id"].unique():
#            pbar.update(1)
#            ind_temp = []
#            # get dataframe of pedestrian i
#            df_temp = df[df["unique_id"] == i]     
#            curr_frame = df_temp["frame"].max()
#            # collect frame at every timestep
#            while curr_frame >= df_temp["frame"].min():
#                try:
#                    ind_temp.append(np.max(df_temp[(df_temp["frame"] <= curr_frame) & (df_temp["frame"] > curr_frame - #timestep)].index.tolist()))
#                except ValueError:
#                    pass
#                curr_frame-=timestep
#            # retain sequence if its length is satisfied
#            if(len(ind_temp) >= min_obs_len):
#                ind = ind + ind_temp
#                debug_op.write("Keeping "+str(df_temp["folderpath"].iloc[-1])+" that has been truncated from "+str(df_temp["frame"].max()-#df_temp["frame"].min()+1)+" frames to "+str(len(ind_temp))+" frames\n")
#            else:
#                debug_op.write("Removing "+str(df_temp["folderpath"].iloc[-1])+" that has been truncated from #"+str(df_temp["frame"].max()-df_temp["frame"].min()+1)+" frames to "+str(len(ind_temp))+" frames\n")
#        ind = sorted(ind)
#        df = df.iloc[ind].reset_index(drop=True)
#        df["unique_id"] = df.groupby(['folderpath']).ngroup()
#               
#        # finalize dataframe 
#        df = df.groupby("unique_id").agg(lambda x: list(x)) 
#        
#        # optional. for debugging purposes
#        # view total number of crossing / non-crossing scenarios
#        #debug_op.write("\n")
#        #crossers = 0
#        #noncrossers = 0
#        #for i in range(len(df)):
#        #    df_temp = df.iloc[i]
#        #    if(df_temp["crossing"][-1] == 1):
#        #        crossers+=1
#        #        debug_op.write("crosser "+str(df_temp["folderpath"][-1])+"\n")
#        #        debug_op.write("standing "+str(df_temp["standing"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("looking "+str(df_temp["looking"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("walking "+str(df_temp["walking"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("crossing "+str(df_temp["crossing"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("\n")
#        #    else:
#        #        noncrossers+=1
#        #        debug_op.write("non-crosser "+str(df_temp["folderpath"][-1])+"\n")
#        #        debug_op.write("standing "+str(df_temp["standing"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("looking "+str(df_temp["looking"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("walking "+str(df_temp["walking"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("crossing "+str(df_temp["crossing"][-1*max_obs_len:])+"\n")
#        #        debug_op.write("\n")
#        
#        #debug_op.write("no crossers: "+str(crossers)+"\n")
#        #debug_op.write("no noncrossers: "+str(noncrossers)+"\n")
#        #debug_op.close()
#        #print("no crossers: ", crossers)
#        #print("no noncrossers: ", noncrossers)   
#        #display(df.iloc[0:40])
#        return df
    
# =============================================================================
def JAADDataset(data_dir, min_obs_len=10, max_obs_len=10, timestep=1):
                        
        #debug_op = open("debug_op.txt", "w")
        # read annotations  
        df = pd.DataFrame()
        print("Reading annotations from ", data_dir)
        pbar = tqdm(total = len(glob.glob(os.path.join(data_dir,"annotations","*"))))
        for file in glob.glob(os.path.join(data_dir,"annotations","*")):
            pbar.update(1)
            df = df.append(pd.read_csv(file), ignore_index=True)  
           
        # assign unique id to each pedestrian
        df["unique_id"] = df.groupby(['folderpath']).ngroup()   
            
        # keep only those that interacted with the driver
        df = df[df["type"] == "pedestrian"].reset_index(drop=True) 
        
        # keep only those that are not under occlusion
        df = df[df["occlusion"] == 0].reset_index(drop=True)      
            
        # --------------------------------------------------------------
        # retain sequence for each pedestrian up till he begins to cross 
        ind = []
        print(" Processing sequence")
        pbar = tqdm(total = int(df["unique_id"].nunique()))
        for i in df["unique_id"].unique():
            pbar.update(1)
            ind_temp = []
            df_temp = df[df["unique_id"] == i]  
            #for folderpath, filename in zip(df_temp["folderpath"][-1*min_obs_len:], df_temp["filename"][-1*min_obs_len:]):
            #    print(os.path.join(folderpath,filename))
            #print(len(np.where(df_temp['crossing']>0)[0]))
            #print(len(df_temp[df_temp['crossing']==1]))
            #print(df_temp['crossing'])
            # pedestrian interacts with driver but does not cross  
            if((len(np.where(df_temp['crossing_true']>0)[0])==0) and (len(df_temp[df_temp['crossing_true']==1])==0)):
                ind_temp = list(df_temp.index.values)
                ind = ind + ind_temp
                #print(ind_temp)
                continue
                                            
            # retain the indices up till pedestrian begins to cross
            ind_temp = list(df_temp.iloc[0:np.min(np.where(df_temp['crossing_true']>0))+1].index.values)
            ind = ind + ind_temp
            #if(len(ind_temp) >= min_obs_len): # remove when uncommenting below
            #    ind = ind + ind_temp
        df = df.iloc[ind].reset_index(drop=True)
        df["unique_id"] = df.groupby(['folderpath']).ngroup() 
                    
        # --------------------------------------------------------------
        # get pedestrian frames in steps of 'timestep'
        # lousier than newer implementation since it takes the min, potentially missing out the final crossing decision
        #ind = []
        #print(" Processing sequence")
        #pbar = tqdm(total = int(df["unique_id"].nunique()))
        #for i in df["unique_id"].unique():
        #    pbar.update(1)
        #    ind_temp = []
        #    # get dataframe of pedestrian i
        #    df_temp = df[df["unique_id"] == i]     
        #    curr_frame = df_temp["frame"].min()-1
        #   # collect frame at every timestep
        #   while curr_frame <= df_temp["frame"].max():
        #        try:
        #            ind_temp.append(np.min(df_temp[(df_temp["frame"] >= curr_frame) & (df_temp["frame"] < curr_frame + timestep)].index.tolist()))
        #        except ValueError:
        #            pass
        #        curr_frame+=timestep
        #    # retain sequence if its length is satisfied
        #    if(len(ind_temp) >= min_obs_len):
        #        ind = ind + ind_temp
        #df = df.iloc[ind].reset_index(drop=True)
        #df["unique_id"] = df.groupby(['folderpath']).ngroup() 
        
        # --------------------------------------------------------------
        # get pedestrian frames in steps of 'timestep'
        # better than older implementation since it takes the max,making it impossible to miss out the final crossing decision
        # need to sort the final indices
        ind = []
        print(" Processing sequence")
        pbar = tqdm(total = int(df["unique_id"].nunique()))
        for i in df["unique_id"].unique():
            pbar.update(1)
            ind_temp = []
            # get dataframe of pedestrian i
            df_temp = df[df["unique_id"] == i]     
            curr_frame = df_temp["frame"].max()
            # collect frame at every timestep
            while curr_frame >= df_temp["frame"].min():
                try:
                    ind_temp.append(np.max(df_temp[(df_temp["frame"] <= curr_frame) & (df_temp["frame"] > curr_frame - timestep)].index.tolist()))
                except ValueError:
                    pass
                curr_frame-=timestep
            # retain sequence if its length is satisfied
            if(len(ind_temp) >= min_obs_len):
                ind = ind + ind_temp
                debug_op.write("Keeping "+str(df_temp["folderpath"].iloc[-1])+" that has been truncated from "+str(df_temp["frame"].max()-df_temp["frame"].min()+1)+" frames to "+str(len(ind_temp))+" frames\n")
            else:
                debug_op.write("Removing "+str(df_temp["folderpath"].iloc[-1])+" that has been truncated from "+str(df_temp["frame"].max()-df_temp["frame"].min()+1)+" frames to "+str(len(ind_temp))+" frames\n")
        ind = sorted(ind)
        df = df.iloc[ind].reset_index(drop=True)
        df["unique_id"] = df.groupby(['folderpath']).ngroup() 
    
        # finalize dataframe 
        df = df.groupby("unique_id").agg(lambda x: list(x)) 
        
        # optional. for debugging purposes
        # view total number of crossing / non-crossing scenarios
        debug_op.write("\n")
        crossers = 0
        noncrossers = 0
        for i in range(len(df)):
            df_temp = df.iloc[i]
            if(df_temp["crossing_true"][-1] == 1):
                crossers+=1
                debug_op.write("crosser "+str(df_temp["folderpath"][-1])+"\n")
                debug_op.write("standing "+str(df_temp["standing"][-1*max_obs_len:])+"\n")
                debug_op.write("looking "+str(df_temp["looking"][-1*max_obs_len:])+"\n")
                debug_op.write("walking "+str(df_temp["walking"][-1*max_obs_len:])+"\n")
                debug_op.write("crossing "+str(df_temp["crossing_true"][-1*max_obs_len:])+"\n")
                debug_op.write("\n")
            else:
                noncrossers+=1
                debug_op.write("non-crosser "+str(df_temp["folderpath"][-1])+"\n")
                debug_op.write("standing "+str(df_temp["standing"][-1*max_obs_len:])+"\n")
                debug_op.write("looking "+str(df_temp["looking"][-1*max_obs_len:])+"\n")
                debug_op.write("walking "+str(df_temp["walking"][-1*max_obs_len:])+"\n")
                debug_op.write("crossing "+str(df_temp["crossing_true"][-1*max_obs_len:])+"\n")
                debug_op.write("\n")
        
        debug_op.write("no crossers: "+str(crossers)+"\n")
        debug_op.write("no noncrossers: "+str(noncrossers)+"\n")
        debug_op.close()
        print("no crossers: ", crossers)
        print("no noncrossers: ", noncrossers)                  
        return df
        
# =============================================================================
class JAADLoader(Dataset):
    def __init__(
            self, df, data_dir, dtype, max_obs_len=15
            ):

        super(JAADLoader, self).__init__()

        self.df = df.copy()
        self.data_dir = data_dir
        self.dtype = dtype
        self.max_obs_len = max_obs_len

        if(self.dtype == "train"):
            self.transform = train_transforms
        if(self.dtype == "val"):
            self.transform = val_transforms

    def __len__(self):
        return len(self.df)

    # -------------------------------
    # retrieves one sample
    def __getitem__(self, index):

        df = self.df.iloc[index]
                
        # load the images, the label, and the relevant filename
        standing  = df["standing"][-1*self.max_obs_len:]              
        looking   = df["looking"][-1*self.max_obs_len:]
        walking   = df["walking"][-1*self.max_obs_len:]
        crossing  = df["crossing_true"][-1*self.max_obs_len:]
                
        pedestrian_images = []
        pedestrian_folderpaths = []
        pedestrian_filenames = []
        for folderpath, filename in zip(df["folderpath"][-1*self.max_obs_len:], df["filename"][-1*self.max_obs_len:]):
            pedestrian_images.append(Image.open(os.path.join(self.data_dir,folderpath,filename)))
            pedestrian_folderpaths.append(folderpath)
            pedestrian_filenames.append(filename)

        # transform
        pedestrian_images = self.transform(pedestrian_images)
        pedestrian_images = torch.stack(pedestrian_images, 0)
        
        return [pedestrian_images, standing, looking, walking, crossing, pedestrian_folderpaths, pedestrian_filenames]
    # -------------------------------

class LausanneLoader(Dataset):
    def __init__(
            self, df, data_dir, dtype, max_obs_len=8
            ):

        super(LausanneLoader, self).__init__()

        self.df = df.copy()
        self.data_dir = data_dir
        self.dtype = dtype
        self.max_obs_len = 8 #max_obs_len # just set to a very high number (1000) if using entire sequence
        print("Lausanne Loader")
        
        if(self.dtype == "train"):
            self.transform = train_transforms
        if(self.dtype == "val"):
            self.transform = val_transforms

    def __len__(self):
        return len(self.df)

    # -------------------------------
    # retrieves one sample
    def __getitem__(self, index):
        
        if(isinstance(index, int)):
            df = self.df.iloc[index]

        else:
            # get the row indexed by index
            df = self.df.iloc[np.asscalar(index.numpy())]
        
        # load the images,
        # the label, and the relevant filename (to fill up the prediction column)      
        pedestrian_images = []
        pedestrian_folderpaths = []
        pedestrian_filenames = []
        i = 0
        
        a = df["folderpath"][-1*self.max_obs_len:]
        a.reverse()
        b = df["filename"][-1*self.max_obs_len:]
        b.reverse()
        
        for folderpath, filename in zip(a, b):
            if(os.path.isfile(os.path.join(self.data_dir,folderpath,filename))):
                pedestrian_images.append(Image.open(os.path.join(self.data_dir,folderpath,filename)))
                pedestrian_folderpaths.append(folderpath)
                pedestrian_filenames.append(filename)
                i=i+1
                if(i==8):
                    break
            else:
                print(os.path.isfile(os.path.join(self.data_dir,folderpath,filename)), " missing")
        
        #for folderpath, filename in zip(df["folderpath"][-1*self.max_obs_len:], df["filename"][-1*self.max_obs_len:]):
        #    if(os.path.isfile(os.path.join(self.data_dir,folderpath,filename))):
        #        pedestrian_images.append(Image.open(os.path.join(self.data_dir,folderpath,filename)))
        #        pedestrian_folderpaths.append(folderpath)
        #        pedestrian_filenames.append(filename)
        #        i=i+1
        #        if(i==8):
        #            break
        #    else:
        #        print(os.path.isfile(os.path.join(self.data_dir,folderpath,filename)), " missing")
        
        pedestrian_images.reverse()
        pedestrian_folderpaths.reverse()
        pedestrian_filenames.reverse()
        
        # transform
        pedestrian_images = self.transform(pedestrian_images)
        pedestrian_images = torch.stack(pedestrian_images, 0)

        return [pedestrian_images, df["crossing_true"][-1], pedestrian_folderpaths, pedestrian_filenames]
    # -------------------------------

# -------------------------------
# transforms for the training set
def train_transforms(images):

    #w=224
    #h=224

    # Resize
    # vgg16 - 120,50
    images = [TF.resize(im, size=(120,50)) for im in images] 

    # Random crop
    # vgg16 - 100,40
    i, j, h, w = transforms.RandomCrop.get_params(images[0], output_size=(100,40)) # 100, 40
    images = [TF.crop(im,i,j,h,w) for im in images]

    # Horizontal flip
    if(random.random() > 0.5):
        images = [TF.hflip(im) for im in images]

    # To Tensor
    images = [TF.to_tensor(im) for im in images]

    # Normalize
    images = [TF.normalize(im, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)) for im in images]

    return images
# -------------------------------

# -------------------------------
# transforms for the validation set
def val_transforms(images):
    
    #images = [TF.pad(im, (0,int((np.shape(im)[0] - np.shape(im)[1])/2))) for im in images]

    # Resize
    # vgg16 - 100,40
    images = [TF.resize(im, size=(100,40)) for im in images]

    # To Tensor
    images = [TF.to_tensor(im) for im in images]

    # Normalize
    images = [TF.normalize(im, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)) for im in images]

    return images
# -------------------------------
