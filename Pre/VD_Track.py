#coding=utf-8
import os
import glob
import dlib
from collections import deque
from skimage import io, transform
import numpy as np
import cv2
import sys
sys.path.append('..')
import Utils.utils as utils
from Posture import *


class VD_Track(object):
    """docstring for VD_Preprocess"""
    def __init__(self, dataset_root, dataset_confs, model_confs=None):
        super(VD_Track, self).__init__()
        self.dataset_folder = os.path.join(dataset_root, 'VD', 'videos')
        self.K_players = dataset_confs.num_players
        self.action_list = dataset_confs.action_list
        self.activity_list = dataset_confs.activity_list
        self.trainval_videos = dataset_confs.splits['trainval']
        self.test_videos = dataset_confs.splits['test']
        self.num_videos = 55

        self.tracker = dlib.correlation_tracker()
        #self.win = dlib.image_window()
        self.save_folder = os.path.join(dataset_root, 'VD', 'person_imgs')
        self.joints_dict = {}
        # track the persons
        self.getPersons()

        # write the train_test file
        #self.getTrainTest()

    def track(self, person_rects, imgs, tracker, save_path):

        for i, person_rect in enumerate(person_rects):
            for j, phase in enumerate(['pre', 'back']):
                if j == 0:
                    j = -1
                for k, f in enumerate(imgs[phase]):
                    #print("Processing Frame {}".format(k))
                    #frame_img = io.imread(f)
                    #print f
                    frame_img = cv2.imread(f)
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

                    if k == 0:
                        x, y, w, h, label, group_label = person_rect
                        #print x,y,w,h
                        tracker.start_track(frame_img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
                    else:
                        tracker.update(frame_img)
                    # save imgs
                    #cropped_image = frame_img[Y:Y + HEIGHT,X + biasRate*WIDTH:X + (1-biasRate)*WIDTH,:]
                    pos = tracker.get_position()
                    top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
                    cropped_image = frame_img[top:bottom,left:right]
                    cropped_image = transform.resize(np.ascontiguousarray(cropped_image),(256,256),mode='constant')

                    img_name = os.path.join(save_path, "%04d_%d_%d.jpg"%(10*i+(5+j*k), label, group_label))
                    #print img_name
                    io.imsave(img_name, cropped_image)


    def getPersons(self):
        # Create the correlation tracker - the object needs to be initialized
        # before it can be used
        for video_id in range(28,55):
            self.joints_dict = {}
            video_id = str(video_id)
            annotation_file = os.path.join(self.dataset_folder, video_id, 'annotations.txt')
            f = open(annotation_file)
            lines = f.readlines()
            imgs={}
            for line in lines:
                frame_id, rects = utils.annotation_parse(line, self.action_list, self.activity_list)
                img_list = sorted(glob.glob(os.path.join(self.dataset_folder, video_id, frame_id, "*.jpg")))[16:26]
                imgs['pre'] = img_list[:5][::-1]
                imgs['back'] = img_list[4:]

                if len(rects)<=self.K_players:
                    print video_id, frame_id
                    save_path = os.path.join(self.save_folder, video_id, frame_id)
                    if not(os.path.exists(save_path)):
                        os.makedirs(save_path)
                        # We will track the frames as we load them off of disk
                        self.track(rects, imgs, self.tracker, save_path)
            utils.save_pkl(self.joints_dict, os.path.join(self.save_folder, 'pose_'+video_id))

    def getTrainTest(self):

        # split train-test following [CVPR 16]
        dataset_config={'trainval':self.trainval_videos,'test':self.test_videos}
        dataList={'trainval':[],'test':[]}
        activity_label_list = {'trainval':[],'test':[]}

        print 'trainval_videos:', dataset_config['trainval']
        print 'test_videos:', dataset_config['test']

        for phase in ['trainval','test']:
            action_list = []
            activity_list = []
            for idx in dataset_config[phase]:
                video_path = self.save_folder  + str(idx) + '/'
                for root, dirs, files in os.walk(video_path):
                    activity_label = ''
                    if len(files)!=0:
                        files.sort()
                        for i in xrange(self.K_players*self.T):
                            if i<len(files):
                                # parse
                                filename = files[i]
                                action_label = filename.split('_')[1]
                                activity_label = filename.split('_')[2].split('.')[0]
                                content_str = root + '/' + filename + '\t' + action_label + '\n'
                                action_list.append(content_str)
                                content_str = root + '/' + filename + '\t' + activity_label + '\n'
                                activity_list.append(content_str)
                            else:
                                # add none.jpg
                                filename = self.save_folder + 'none.jpg'
                                content_str = filename + '\t' + 'error' + '\n'
                                action_list.append(content_str)
                                content_str = filename + '\t' + activity_label + '\n'
                                activity_list.append(content_str)


            self.write_list(action_list, self.T, '_action', phase)
            self.write_list(activity_list, self.T*self.K_players, '_activity', phase)


    def write_list(self, source_list, block_size, label_type, phase):
        source_list = utils.block_shuffle(source_list, block_size)
        txtFile = self.save_folder + phase + label_type + '.txt'
        open(txtFile, 'w')
        print phase +'_size:' + str(len(source_list)/(block_size))
        for i in range(len(source_list)):
            with open(txtFile, 'a') as f:
                f.write(source_list[i])
