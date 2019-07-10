import os, shutil
import glob
import numpy as np
import cv2
import sys
sys.path.append('..')
import utils


class Rank_MI(object):
    """docstring for Rank_MI"""
    def __init__(self, dataset_root, dataset_name, dataset_confs, model_confs):
        super(Rank_MI, self).__init__()
        # Path to the video frames
        self.imgs_folder = os.path.join(dataset_root, dataset_name, 'imgs')
        self.ranked_folder = os.path.join(dataset_root, dataset_name, 'imgs_ranked')
        self.num_groups = model_confs.num_groups
        self.num_videos = dataset_confs.num_videos
        #print self.imgs_folder, self.ranked_folder
        self.rank()
        

    def rank(self):
        for video_id in xrange(self.num_videos):
            video_folder = os.path.join(self.imgs_folder, str(video_id))
            for root, dirs, files in os.walk(video_folder):
                if len(files)!=0:
                    keys = root.split('/')
                    clips_folder = root
                    save_folder = os.path.join(self.ranked_folder, keys[-2], keys[-1])
                    if not os.path.exists(save_folder):
                        print clips_folder, save_folder
                        self.ranking(clips_folder, save_folder, N_g = self.num_groups)
                    else:
                        print "%s exist!" % (save_folder)


    def calc_Motion_Intensity(self, imgs):
        Sum_stack_flow = np.zeros([256,256,1])
        for i in xrange(len(imgs)-1):
            frame1 = cv2.imread(imgs[i])
            frame2 = cv2.imread(imgs[i+1])
            pre = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            cur = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(pre, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            Sum_stack_flow[...,0] += abs(flow[...,0])
            Sum_stack_flow[...,0] += abs(flow[...,1])
        Motion_Intensity = np.sum(Sum_stack_flow)/(256*256)
        return Motion_Intensity

    def copy_files(self, srcfile, targetfile):
        if not os.path.isfile(srcfile):
            print "%s not exist!" % (srcfile)
        else:
            path, name = os.path.split(targetfile)
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copyfile(srcfile, targetfile)


    def ranking(self, clips_folder, save_folder, N_g = None):
        T = 10
        idx = 1
        imgs = glob.glob(os.path.join(clips_folder, "*.jpg"))
        imgs.sort()
        Motion_Intensity = np.zeros(len(imgs)/10)
        Ranked_idx = np.zeros(len(imgs)/10, dtype = int)
        for k in xrange(len(imgs)/10):
            Motion_Intensity[k] = self.calc_Motion_Intensity(imgs[k*10:(k+1)*10])
        
        # rank motion_intensity
        if N_g:
            K = 12
            for g in xrange(N_g):
                S_g = g*(K/N_g)
                E_g = (g+1)*(K/N_g)
                if E_g > len(imgs)/10:
                    E_g = len(imgs)/10
                part_MI = Motion_Intensity[S_g:E_g]
                temp = sorted(range(len(part_MI)),key=lambda k:part_MI[k])
                temp.reverse()
                Ranked_idx[S_g:E_g] = np.asarray(temp)+S_g
        else:
            Ranked_idx = sorted(range(len(Motion_Intensity)),key=lambda k:Motion_Intensity[k])
            Ranked_idx.reverse()

        # rewrite imgs by the ranked_idx
        print 'copying %s'%(clips_folder)
        for k in Ranked_idx:
            for t in xrange(T):
                src_img = imgs[k*10+t]
                action = int(src_img.split('/')[-1].split('_')[-2])
                activity = int(src_img.split('/')[-1].split('_')[-1].split('.')[0])
                target_img = os.path.join(save_folder,"%04d_%d_%d.jpg"%(idx, action, activity))
                idx +=1
                self.copy_files(src_img, target_img)
