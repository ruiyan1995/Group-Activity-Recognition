#coding=utf-8
import os
import glob
import utils
from Track import *
from Rank_MI import *

class VD_Track(Track):
    """docstring for VD_Preprocess"""
    def __init__(self, dataset_root, dataset_confs, model_confs=None, ranked=False):
        super(VD_Track, self).__init__(dataset_root, dataset_confs, 'VD', model_confs)
        # track the persons
        self.getPersons()
        # rank by MI
        if ranked:
            self.save_folder = os.path.join(dataset_root, 'VD', 'imgs_ranked')
            Rank_MI(dataset_root, 'VD', dataset_confs, model_confs)
        # write the train_test file
        self.getTrainTest()


    def getPersons(self):
        # Create the correlation tracker - the object needs to be initialized
        # before it can be used
        for video_id in range(self.num_videos):
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
                
                if len(rects)<=self.num_players:
                    #print video_id, frame_id
                    save_path = os.path.join(self.save_folder, video_id, frame_id)
                    
                    if not(os.path.exists(save_path)):
                        os.makedirs(save_path)
                    # We will track the frames as we load them off of disk
                    self.track(rects, imgs, self.tracker, save_path)
        

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
