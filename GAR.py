"""
	Group Activity Recognition
"""

import argparse
import Runtime
import torch
import Pre
import time

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='./dataset/', help='Please set the root folder of datasets')
parser.add_argument('--dataset_name', type=str, default='VD', choices=['VD', 'CAD'], help='Please choose one of the dataset')
#parser.add_argument('--mode', type=str, default='extract_action_feas', choices=['trainval_action', 'extract_action_feas', 'frame_trainval_activity', 'trainval_activity'], help='Please choose mode')

opt = parser.parse_args()


# Step Zero: Dataset Preprocessing
print 'Please wait for tracking! about 240min for VD'
track_since = time.time()
Pre.Processing(opt.dataset_root, opt.dataset_name, 'track')
print('Tracking {} in {:.0f}m {:.0f}s'.format(opt.dataset_name,
            (time.time() - track_since) // 60, time_elapsed % 60))

print 'Please wait for ranking!  about 180min for VD'
rank_since = time.time()
Pre.Processing(opt.dataset_root, opt.dataset_name, 'rank')
print('Ranking {} in {:.0f}m {:.0f}s'.format(opt.dataset_name,
            (time.time() - rank_since) // 60, (time.time() - rank_since) % 60))


# Step One: action recognition
print 'Please wait for training action! Needs 200min for 20epochs(VD).'
trainval_action_since = time.time()
Action = Runtime.Action_Level(opt.dataset_root, opt.dataset_name, 'trainval_action')
Action.trainval()
print('Training action {} in {:.0f}m {:.0f}s'.format(opt.dataset_name,
            (time.time() - trainval_action_since) // 60, (time.time() - trainval_action_since) % 60))


print 'Please wait for extracting action_feas! '
extract_since = time.time()
Action = Runtime.Action_Level(opt.dataset_root, opt.dataset_name, 'extract_action_feas')
Action.extractFeas()
print('Extracting action_feas {} in {:.0f}m {:.0f}s'.format(opt.dataset_name,
            (time.time() - extract_since) // 60, (time.time() - extract_since) % 60))


# Step Two: Group activity recognition
Activity = Runtime.Activity_Level(opt.dataset_root, opt.dataset_name, 'trainval_activity')
Activity.trainval()
