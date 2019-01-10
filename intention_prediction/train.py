import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# for debug
import cv2
import numpy as np
import matplotlib.pyplot as plt
# for debug

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss

from sgan.models import CNNLSTM, CNNMP
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset', default='./datasets/lausanne', type=str)
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--timestep', default=30, type=int)
parser.add_argument('--obs_len', default=8, type=int)

# Optimization
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--h_dim', default=32, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=64, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    # Build the training set
    logger.info("Initializing train set")
    train_path = os.path.join(args.dataset, "train")
    train_dset, train_loader = data_loader(args, train_path, "train")
        
    # Build the validation set
    logger.info("Initializing val set")
    val_path = os.path.join(args.dataset, "val") 
    val_dset, val_loader = data_loader(args, val_path, "val")

    # set data type to cpu/gpu
    long_dtype, float_dtype = get_dtypes(args)

    # Build train val dataset
    #trainval_path = os.path.join(os.getcwd(), "dataset")
    #logger.info("Initializing train-val dataset")
    #train_dset, train_loader, _, val_loader = data_loader(args, trainval_path)
   
    iterations_per_epoch = train_dset / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    # initialize the CNN LSTM
    classifier = CNNLSTM(
            embedding_dim=args.embedding_dim,
            h_dim=args.h_dim,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout)
    classifier.apply(init_weights)
    classifier.type(float_dtype).train()
    #input()    

    #classifier = CNNMP(
    #        no_filters=32)
    #classifier.apply(init_weights)
    #classifier.type(float_dtype).train()

    # set the optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

    # define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        classifier.load_state_dict(checkpoint['classifier_state'])
        optimizer.load_state_dict(checkpoint['classifier_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'classifier_losses': defaultdict(list), # classifier loss
            'losses_ts': [],                        # loss at timestep ?
            'metrics_val': defaultdict(list),       # valid metrics (loss and accuracy)
            'metrics_train': defaultdict(list),     # train metrics (loss and accuracy)
            'sample_ts': [],
            'restore_ts': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'classifier_state': None,
            'classifier_optim_state': None,
            'classifier_best_state': None,
            'best_t': None,
        }
    t0 = None
    print("Total no of iterations: ", args.num_iterations)
    while t < args.num_iterations:
                
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))

        for batch in train_loader:

            # Maybe save a checkpoint
            if t == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, classifier, loss_fn)
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, classifier, loss_fn)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_loss = min(checkpoint['metrics_val']['d_loss'])
                max_acc  = max(checkpoint['metrics_val']['d_accuracy'])

                if metrics_val['d_loss'] == min_loss:
                    logger.info('New low for data loss')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                if metrics_val['d_accuracy'] == max_acc:
                    logger.info('New high for accuracy')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['classifier_state'] = classifier.state_dict()
                checkpoint['classifier_optim_state'] = optimizer.state_dict()
                checkpoint_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

            #(images, labels) = batch
            # reference
            #print("batch size ", len(images))			                # batch size (total no. of sequences where each sequence can have diff. no. of images)
            #print("sequence length for sample[0] ", len(images[0]))		# number of images for sample 0
            #print("sequence length for sample[1] ", len(images[1]))
            #print("sequence length for sample[2] ", len(images[2]))
            #print("size of first image for sample[0] ", np.shape(images[0][0]))	# size of first image of sample 0

            # measure time between batches
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()
        
            # run batch and get losses
            losses = step(args, batch, classifier, loss_fn, optimizer)

            # measure time between batches
            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # measure time between batches
            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, classifier, loss_fn)
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, classifier, loss_fn)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_loss = min(checkpoint['metrics_val']['d_loss'])
                max_acc  = max(checkpoint['metrics_val']['d_accuracy'])

                if metrics_val['d_loss'] == min_loss:
                    logger.info('New low for data loss')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                if metrics_val['d_accuracy'] == max_acc:
                    logger.info('New high for accuracy')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['classifier_state'] = classifier.state_dict()
                checkpoint['classifier_optim_state'] = optimizer.state_dict()
                checkpoint_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                #checkpoint_path = os.path.join(args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                #logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                #key_blacklist = [
                #    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                #    'g_optim_state', 'd_optim_state', 'd_best_state',
                #    'd_best_nl_state'
                #]
                #small_checkpoint = {}
                #for k, v in checkpoint.items():
                #    if k not in key_blacklist:
                #        small_checkpoint[k] = v
                #torch.save(small_checkpoint, checkpoint_path)
                #logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                                
                # print best 
                #print("[train] best accuracy ", checkpoint[]
                print("[train] best accuracy at lowest loss ",checkpoint['metrics_train']['d_accuracy'][np.argmin(checkpoint['metrics_train']['d_loss'])])
                print("[train] best accuracy at highest accuracy ", max(checkpoint['metrics_train']['d_accuracy']))
                print("[val] best accuracy at lowest loss ",  checkpoint['metrics_val']['d_accuracy'][np.argmin(checkpoint['metrics_val']['d_loss'])])
                print("[val] best accuracy at highest accuracy ", max(checkpoint['metrics_val']['d_accuracy']))                

                break

def step(args, batch, classifier, loss_fn, optimizer):
    
    (pedestrian_crops, decision_true, _) = batch

    losses = {}
    loss = torch.zeros(1).type(torch.cuda.FloatTensor)

    # predict pedestrian decision
    decision_pred = classifier(pedestrian_crops)

    # compute loss
    data_loss = loss_fn(decision_pred, decision_true.cuda())
    
    # record loss at current batch and total loss
    losses['data_loss'] = data_loss.item()
    loss += data_loss
    losses['total_loss'] = loss.item()

    # backprop given the loss
    optimizer.zero_grad()
    loss.backward()
    #if args.clipping_threshold > 0:
    #    nn.utils.clip_grad_norm_(classifier.parameters(),args.clipping_threshold)
    optimizer.step()

    return losses

def guided_backprop(
    args, loader, classifier,
):
    data_confusions = []
    metrics = {}
    #classifier.eval()
    
    for batch in loader:
        
        # get batch
        (pedestrian_crops, decision_true, _) = batch
          
        print("batch size ", len(pedestrian_crops))
        print("timesteps",  len(pedestrian_crops[0]))
       
        # predict decision
        decision_pred = classifier(pedestrian_crops, input_as_var=True)
        onehot_pred = torch.round(decision_pred.cpu())

        #print(classifier.gradients)

        # backprop
        classifier.zero_grad()
        decision_pred.backward(gradient=onehot_pred.cuda())

        input()

    #classifier.train()
    return metrics

def check_accuracy(
    args, loader, classifier, loss_fn
):
    data_losses = []
    data_confusions = []
    metrics = {}
    classifier.eval()
    with torch.no_grad():
        for batch in loader:
        
            # get batch
            (pedestrian_crops, decision_true, _) = batch
            
            # predict decision
            decision_pred = classifier(pedestrian_crops)
            
            # print(decision_pred) 
            # tensor([[-3.2049e+00,  3.0137e+00],
            #         [ 7.4607e+00, -6.7692e+00],
            #         [ 8.2550e-01, -1.0655e+00],
            #         [ 6.7967e+00, -6.2332e+00],
            #         [ 6.7422e+00, -6.1310e+00],
            #         [-4.0681e+00,  3.5863e+00],
            #         [-2.7053e-03,  8.3053e-01],
            #         [ 1.0868e+00, -4.2169e-01],
            #         [-1.4344e+00,  1.0542e+00],
            #         [ 9.5990e+00, -8.6439e+00],
            #         [ 2.0644e+00, -1.7798e+00],
            #         [ 7.9568e+00, -7.5459e+00],
            #         [-1.8810e-01,  6.8803e-01],
            #         [ 6.6334e+00, -5.6960e+00],
            #         [-5.8433e+00,  5.9706e+00],
            #         [-2.7829e+00,  2.7846e+00]], device='cuda:0')
               
            # print(decision_pred.max(1))
            # (tensor([3.0137,
            #          7.4607, 
            #          0.8255, 
            #          6.7967, 
            #          6.7422, 
            #          3.5863, 
            #          0.8305, 
            #          1.0868, 
            #          1.0542,
            #          9.5990,  
            #          2.0644, 
            #          7.9568, 
            #          0.6880, 
            #          6.6334, 
            #          5.9706, 
            #          2.7846], device='cuda:0'),
            # tensor([1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], device='cuda:0'))
            
            # compute loss
            data_loss = loss_fn(decision_pred, decision_true.cuda())
            data_losses.append(data_loss.item())

            # compute accuracy
            #print(decision_pred)
            #print(decision_pred.max(1)[1])
            #print(decision_true)
            #print(torch.eq(decision_true, decision_pred.max(1)[1].cpu()))

            # build confusion matrix
            data_confusion = confusion_matrix(decision_true.numpy(), decision_pred.max(1)[1].cpu().numpy())
            data_confusions.append(data_confusion)

            # compute accuracy
            #data_accuracy = float(torch.eq(decision_true, decision_pred.max(1)[1].cpu()).sum())/float(decision_true.numel())
            #data_accuracies.append(data_accuracy)
            
            #print("true ", decision_true)      
            #print("pred ", decision_pred.max(1)[1].cpu())
            #print("pred ", decision_pred)
            #print("sum  ", torch.eq(decision_true, decision_pred.max(1)[1].cpu()))
            #print("sum  ", float(torch.eq(decision_true, decision_pred.max(1)[1].cpu()).sum()))
            #print(data_accuracy)

    #verify confusion matrix
    #s = 0
    #ss = 0
    #for a,b, in zip(data_accuracies, data_len):
    #    s = s + a*b
    #    ss = ss + b
    #s = float(s) / float(ss)
    #print(s)

    tn, fp, fn, tp = sum(data_confusions).ravel()

    # record metrics
    metrics['d_loss'] = sum(data_losses) / len(data_losses) # not entirely accurate because of the batch size for the final loop
    metrics['d_accuracy']  = (tp + tn) / (tn + fp + fn + tp) # accurate
    metrics['d_precision'] = tp / (tp + fp)
    metrics['d_recall']    = tp / (tp + fn)
    metrics['d_tn'] = tn 
    metrics['d_fp'] = fp
    metrics['d_fn'] = fn
    metrics['d_tp'] = tp

    classifier.train()
    return metrics

def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
