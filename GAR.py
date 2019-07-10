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


opt = parser.parse_args()


# Step Zero: Dataset Preprocessing
print 'Please wait for tracking and ranking! about 240min + 180min'
track_since = time.time()
Pre.Processing(opt.dataset_root, opt.dataset_name, operation='track', ranked=True)
print('Tracking and ranking {} in {:.0f}m {:.0f}s'.format(opt.dataset_name,
            (time.time() - track_since) // 60, (time.time() - track_since) % 60))


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


# Step Two: Evaluate
dataset_size = 1337 if opt.dataset_name == 'VD' else 621
since = time.time()
Activity.evaluate()
print 'infer one sequence, takes', (time.time()-since)/dataset_size, 's'
