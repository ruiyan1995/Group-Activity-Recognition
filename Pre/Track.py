#coding=utf-8
import os
import glob
import dlib
from collections import deque
import numpy as np
import cv2
import utils

class Track(object):
    """docstring for Track"""
    def __init__(self, dataset_root, dataset_confs, dataset_name, model_confs=None):
        super(Track, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_folder = os.path.join(dataset_root, dataset_name, 'videos')
        self.num_players = dataset_confs.num_players
        self.num_videos = dataset_confs.num_videos
        self.num_frames = model_confs.num_frames
        
        self.phases = ['trainval','test']
        self.videos = {phase: dataset_confs.splits[phase] for phase in self.phases}
        
        self.action_list = dataset_confs.action_list
        self.activity_list = dataset_confs.activity_list
        
        self.tracker = dlib.correlation_tracker()
        self.save_folder = os.path.join(dataset_root, dataset_name, 'imgs')
        print 'the person imgs are saved at',self.save_folder

    def annotation_parse(self, line):
        keywords = deque(line.strip().split(' '))
        frame_id = keywords.popleft().split('.')[0]
        activity = self.activity_list.index(keywords.popleft())
        Rects = []
        while keywords:
            x = int(keywords.popleft())
            y = int(keywords.popleft())
            w = int(keywords.popleft())
            h = int(keywords.popleft())
            action = self.action_list.index(keywords.popleft())
            Rects.append([x,y,w,h,action,activity])
        Rects = np.asarray(Rects)
        # sort Rects by the first col
        Rects = Rects[np.lexsort(Rects[:,::-1].T)]
        return frame_id, Rects

    def track(self, person_rects, imgs, tracker, save_path):
        for i, person_rect in enumerate(person_rects):
            for j, phase in enumerate(['pre', 'back']):
                if j == 0:
                    j = -1
                for k, f in enumerate(imgs[phase]):
                    #print("Processing Frame {}".format(k))
                    frame_img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                    if k == 0:
                        x, y, w, h, label, group_label = person_rect
                        #print x,y,w,h
                        tracker.start_track(frame_img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
                    else:
                        tracker.update(frame_img)
                    
                    # save imgs
                    pos = tracker.get_position()
                    top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
                    cropped_image = frame_img[top:bottom,left:right]
                    cropped_image = cv2.resize(cropped_image,(256,256), interpolation=cv2.INTER_CUBIC)

                    img_name = os.path.join(save_path, "%04d_%d_%d.jpg"%(10*i+(5+j*k), label, group_label))
                    cv2.imwrite(img_name, cropped_image)

                    
    def write_list(self, source_list, block_size, label_type, phase):
        source_list = utils.block_shuffle(source_list, block_size)
        txtFile = os.path.join(self.save_folder, phase + '_' + label_type + '.txt')
        open(txtFile, 'w')
        print label_type + '_' + phase +'_size:' + str(len(source_list)/(block_size))
        with open(txtFile, 'w') as f:
            for i in range(len(source_list)):
                f.write(source_list[i])
