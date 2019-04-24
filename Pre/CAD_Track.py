#coding=utf-8
import os
import glob
import numpy as np
import sys
import utils
from Track import *
from Rank_MI import *

class CAD_Track(Track):
    """docstring for CAD_Track"""
    def __init__(self, dataset_root, dataset_confs, model_confs=None, ranked=False):
        super(CAD_Track, self).__init__(dataset_root, dataset_confs, 'CAD', model_confs)
        # modify the annotation_file
        self.modify()

        # track the persons
        self.getPersons()
        
        # rank by MI
        if ranked:
            self.save_folder = os.path.join(dataset_root, 'VD', 'imgs_ranked')
            Rank_MI(dataset_root, 'VD', dataset_confs, model_confs)

        # write the train_test file
        self.getTrainTest()

    def annotation_parse(self, line):
        keywords = deque(line.strip().split(' '))
        frame_id = keywords.popleft().split('.')[0]
        activity = int(keywords.popleft())
        Rects = []
        while keywords:
            x = int(keywords.popleft())
            y = int(keywords.popleft())
            w = int(keywords.popleft())
            h = int(keywords.popleft())
            action = int(keywords.popleft())
            Rects.append([x,y,w,h,action,activity])
        Rects = np.asarray(Rects)
        # sort Rects by the first col
        Rects = Rects[np.lexsort(Rects[:,::-1].T)]
        return frame_id, Rects



    def modify_annotation(self, annotation_file):
        # modify the annotation for CAD, as same as VD
        new_file = annotation_file.split('.txt')[0] + '_new.txt'
        open(new_file, 'w')

        f = open(annotation_file)
        lines = f.readlines()
        cur_frameId = int(lines[0].split('\t')[0])
        content_str = ''
        action_count = np.zeros([5])
        sep = ' '
        num = 0
        for line in lines:
            keywords = line.split('\t')
            frame_id = int(keywords[0])
            action = int(keywords[5])
            if frame_id%10==1 and action!=1:
                action = action-2
                x,y,w,h = int(keywords[1]),int(keywords[2]),int(keywords[3]),int(keywords[4])
                x=0 if x<0 else x
                y=0 if y<0 else y
                if w<=0 or h<=0:
                    print 'error!'
                    break
                anno_str = sep + str(x) + sep + str(y) + sep + str(w) + sep + str(h) + sep + str(action)

                '''if frame_id == cur_frameId:
                    action_label_count[action_label] += 1 
                    content_str = content_str + anno_str'''
                if frame_id != cur_frameId:
                    activity = np.argmax(action_count)
                    content_str = str(cur_frameId) + sep + str(activity) + content_str + '\n'
                    utils.write_txt(new_file, content_str, 'a')
                    num +=1
                    cur_frameId = frame_id
                    content_str = ''
                    action_count = np.zeros([5])

                action_count[action] += 1 
                content_str = content_str + anno_str
        activity = np.argmax(action_count)
        content_str = str(cur_frameId) + sep + str(activity) + content_str + '\n'
        utils.write_txt(new_file, content_str, 'a')
        num +=1
        return num

    def modify(self):
        total_train = 0
        total_test = 0
        for video_id in range(1, self.num_videos+1):
            print video_id
            video_folder = os.path.join(self.dataset_folder, 'seq' + '%02d'%video_id)
            print video_folder
            file = os.path.join(video_folder, 'annotations.txt')
            num_clips = self.modify_annotation(file)
            if video_id in self.trainval_videos:
                total_train += num_clips
            else:
                total_test += num_clips
            print total_train, total_test


    def getPersons(self):
        for video_id in range(1, 10):
            self.joints_dict = {}
            video_folder = os.path.join(self.dataset_folder, 'seq'+ '%02d'%video_id)
            video_id = str(video_id)
            annotation_file = os.path.join(video_folder, 'annotations_new.txt')
            lines = open(annotation_file).readlines()
            img_list = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
            #print video_id, len(img_list)

            for line in lines:
                frame_id, rects = self.annotation_parse(line)
                imgs={}
                if int(frame_id) + 4 <=len(img_list) and int(frame_id)-5>0:
                    print video_id, frame_id
                    clip_list = img_list[int(frame_id)-6:int(frame_id)+4]
                    imgs['pre'] = clip_list[:5][::-1]
                    imgs['back'] = clip_list[4:]
                    #print imgs
                    save_path = os.path.join(self.save_folder, video_id, frame_id)
                    if not(os.path.exists(save_path)):
                        os.makedirs(save_path)
                        # We will track the frames as we load them off of disk
                        self.track(rects, imgs, self.tracker, save_path)
            utils.save_pkl(self.joints_dict, os.path.join(self.save_folder, 'pose_'+video_id))


    def getTrainTest(self):
        dataset_config={'trainval':self.trainval_videos,'test':self.test_videos}
        dataList={'trainval':[],'test':[]}
        activity_label_list = {'trainval':[],'test':[]}

        print 'trainval_videos:', dataset_config['trainval']
        print 'test_videos:', dataset_config['test']
        for phase in ['trainval','test']:
            action_list = []
            for idx in dataset_config[phase]:
                video_path = os.path.join(self.save_folder, str(idx))
                for root, dirs, files in os.walk(video_path):
                    activity_label = ''
                    if len(files)!=0:                       
                        files.sort()
                        for i in xrange(self.num_players*self.num_frames):
                            if i<len(files):
                                # parse
                                filename = files[i]
                                action_label = filename.split('_')[1]
                                activity_label = filename.split('_')[2].split('.')[0]
                                
                                # Merge 'Walking' and 'Crossing' as 'Moving'
                                if int(action_label) == 3:
                                    action_label = str(0)
                                elif int(action_label) == 4:
                                    action_label = str(3)

                                if int(activity_label) == 3:
                                    activity_label = str(0)
                                elif int(activity_label) == 4:
                                    activity_label = str(3)
                                
                                
                                content_str = root + '/' + filename + '\t' + action_label + '\n'
                                action_list.append(content_str)
                                content_str = root + '/' + filename + '\t' + activity_label + '\n'
                                activity_list.append(content_str)
                            else:
                                # add none.jpg
                                filename = os.path.join(self.save_folder, 'none.jpg')
                                content_str = filename + '\t' + 'error' + '\n'
                                action_list.append(content_str)
                                content_str = filename + '\t' + activity_label + '\n'
                                activity_list.append(content_str)
            
            self.write_list(action_list, self.num_frames, 'action', phase)
            self.write_list(activity_list, self.num_frames*self.num_players, 'activity', phase)

