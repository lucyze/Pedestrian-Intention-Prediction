from torch.utils.data import DataLoader
from sgan.data.trajectories import JAADDataset, JAADLoader, JAADcollate

import torch
import numpy as np

def data_loader(args, path, dtype):
   
    # build the train set
    if(dtype == "train"):
        df = JAADDataset(path, args.min_obs_len, args.max_obs_len, args.timestep)
        dataset = JAADLoader(df, path, dtype, args.max_obs_len)
        #print(dtype, " has ", neg_sample_size, " negative samples and ", pos_sample_size, " positive samples")

        # build the train iterator
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            collate_fn=JAADcollate,
            shuffle=True)
            #sampler=sampler)  

    # build the val set
    # validation set should never use the weighted random sampler
    if(dtype == "val"):
        df = JAADDataset(path, args.min_obs_len, args.max_obs_len, args.timestep)
        dataset = JAADLoader(df, path, dtype, args.max_obs_len)
        #print(dtype, " has ", neg_sample_size, " negative samples and ", pos_sample_size, " positive samples")

        # build the val iterator
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            collate_fn=JAADcollate)
   
    # build the training set set
    #if(dtype == "train"):
    #    df, weights, neg_sample_size, pos_sample_size = PrepDataset(path, args.obs_len, args.timestep)
    #    dataset = TrajectoryDataset(df,"train", args.obs_len)
    #    #sampler = WeightedRandomSampler(weights, len(weights))
    #    print(dtype, " has ", neg_sample_size, " negative samples and ", pos_sample_size, " positive samples")

    #    # build the train iterator
    #    loader = DataLoader(
    #        dataset,
    #        batch_size=args.batch_size,
    #        num_workers=1,
    #        collate_fn=seq_collate,
    #        shuffle=True)
    #        #sampler=sampler)

    # build the validation set
    # validation set should not use the weighted random sampler
    #if(dtype == "val"):
    #    df, _, neg_sample_size, pos_sample_size = PrepDataset(path, args.obs_len, args.timestep)
    #    dataset = TrajectoryDataset(df,"val", args.obs_len)
    #    print(dtype, " has ", neg_sample_size, " negative samples and ", pos_sample_size, " positive samples")

    #    # build the val iterator
    #    loader  = DataLoader(
    #        dataset,
    #        batch_size=args.batch_size,
    #        num_workers=1,
    #        collate_fn=seq_collate,
    #        shuffle=True)

    ## build the train iterator
    #train_dataset = TrajectoryDataset(train_df)
    #train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
    #train_loader  = DataLoader(
    #    train_dataset,
    #    batch_size=16,
    #    num_workers=1,
    #    collate_fn=seq_collate,
    #    sampler=train_sampler)
    
    ## build the val iterator
    #valid_dataset = TrajectoryDataset(valid_df)
    #valid_loader  = DataLoader(
    #    valid_dataset,
    #    batch_size=16,
    #    num_workers=1,
    #    collate_fn=seq_collate,
    #    shuffle=True)

    return len(df), loader
