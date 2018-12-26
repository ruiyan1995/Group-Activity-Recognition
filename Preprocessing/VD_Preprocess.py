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
import utils


class VD_Preprocess(object):
	"""docstring for VD_Preprocess"""
	def __init__(self, dataset_root):
		super(VD_Preprocess, self).__init__()
		self.K_players = 12
		self.actionList = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']
		self.activityList = ['l-pass', 'r-pass', 'l_set', 'r_set', 'l-spike', 'r_spike', 'l_winpoint', 'r_winpoint']
		self.trainval_videos = [0,1,2,3,6,7,8,10,12,13,15,16,17,18,19,
							22,23,24,26,27,28,30,31,32,33,36,38,39,
							40,41,42,46,48,49,50,51,52,53,54]
		self.test_videos = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]
		# track the persons
		self.__getPersons(dataset_root)

		# write the train_test file
		self.__getTrainTest(img_folder=os.join(dataset_root, 'imgs'))

	def __annotationParse(self, line):
		keywords = deque(line.strip().split(' '))
		frame_id = keywords.popleft().split('.')[0]
		activity = self.activityList.index(keywords.popleft())
		Rects = []
		while keywords:
			x = int(keywords.popleft())
			y = int(keywords.popleft())
			w = int(keywords.popleft())
			h = int(keywords.popleft())
			action = self.actionList.index(keywords.popleft())
			Rects.append([x,y,w,h,action,activity])
		Rects = np.asarray(Rects)
		# sort Rects by the first col
		Rects = Rects[np.lexsort(Rects[:,::-1].T)]
		return frame_id, Rects

	def __track(self, person_rects, imgs, tracker, win, save_path):
		for i, person in enumerate(person_rects):
			for j, phase in enumerate(['pre', 'back']):
				if j == 0:
					j = -1
				for k, f in enumerate(imgs[phase]):
					#print("Processing Frame {}".format(k))
					img = io.imread(f)
					if k == 0:
						x, y, w, h, label, group_label = person
						#print x,y,w,h
						tracker.start_track(img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
					else:
						tracker.update(img)
					# save imgs
					#cropped_image = img[Y:Y + HEIGHT,X + biasRate*WIDTH:X + (1-biasRate)*WIDTH,:]
					pos = tracker.get_position()
					top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
					cropped_image = img[top:bottom,left:right]
					cropped_image = transform.resize(np.ascontiguousarray(cropped_image),(256,256),mode='constant')
					img_name = os.path.join(save_path, "%04d_%d_%d.jpg"%(10*i+(5+j*k), label, group_label))
					#print img_name
					io.imsave(img_name, cropped_image)
	
	def __getPersons(self, datasetPath, output_folder):
		# Create the correlation tracker - the object needs to be initialized
		# before it can be used
		tracker = dlib.correlation_tracker()
		win = dlib.image_window()
		for video_id in range(55):
			video_id = str(video_id)
			annotation_file = datasetPath + video_id + '/annotations.txt'
			f = open(annotation_file)
			lines = f.readlines()
			imgs={}
			for line in lines:
				frame_id, rects = __annotationParse(line)
				img_list = sorted(glob.glob(os.path.join(datasetPath, video_id, frame_id, "*.jpg")))[16:26]
				imgs['pre'] = img_list[:5][::-1]
				imgs['back'] = img_list[4:]
				
				if len(rects)<=K_players:
					print video_id, frame_id
					save_path = os.path.join(output_folder, video_id, frame_id)
					if not(os.path.exists(save_path)):
						os.makedirs(save_path)
						# We will track the frames as we load them off of disk
						track(rects, imgs, tracker, win, save_path)
	
	def __getTrainTest(self, img_folder):
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
				video_path = img_folder  + str(idx) + '/'
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
								filename = img_folder + 'none.jpg'
								content_str = filename + '\t' + 'error' + '\n'
								action_list.append(content_str)
								content_str = filename + '\t' + activity_label + '\n'
								activity_list.append(content_str)
			
			# write action txt
			action_list = utils.block_shuffle(action_list, self.T)
			txtFile = img_folder + phase + '_action' + '.txt'
			open(txtFile, 'w')
			print phase +'_size:' + str(len(action_list)/(self.T*self.K_players))
			for i in range(len(action_list)):
				with open(txtFile, 'a') as f:
					f.write(action_list[i])

			# write activity txt
			activity_list = utils.block_shuffle(activity_list, self.T*self.K_players)
			txtFile = img_folder + phase + '_activity' + '.txt'
			open(txtFile, 'w')
			print phase +'_size:' + str(len(activity_list)/(self.T*self.K_players))
			for i in range(len(activity_list)):
				with open(txtFile, 'a') as f:
					f.write(activity_list[i])

