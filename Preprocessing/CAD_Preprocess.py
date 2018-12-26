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

class CAD_Preprocess(object):
	"""docstring for CAD_Preprocess"""
	def __init__(self, datasetPath, outputPath):
		super(CAD_Preprocess, self).__init__()
		self.datasetPath = datasetPath
		self.outputPath = outputPath
		self.K_players = 5
		self.T = 10
		self.trainval_videos = [7,12,13,14,15,16,17,18,19,20,
								21,22,23,24,26,27,30,31,32,33,
								34,35,36,37,38,39,40,41,42,43,44]
		self.test_videos = [1,2,3,4,5,6,8,9,10,11,25,28,29]
		# track the persons
		self.__getPersons(datasetPath, output_folder)

		# write the train_test file
		self.__getTrainTest(img_folder=output_folder)

	def __annotationParse(line):
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



	def __modify_annotation(annotation_file):
		# modify the annotation for CAD, as same as VD
		new_file = annotation_file.split('.')[0] + '_new.txt'
		open(new_file, 'w')

		f = open(annotation_file)
		lines = f.readlines()
		cur_frameId = int(lines[0].split('\t')[0])
		content_str = ''
		action_count = np.zeros([Class_Num])
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
					action_count = np.zeros([Class_Num])

				action_count[action] += 1 
				content_str = content_str + anno_str
		activity = np.argmax(action_count)
		content_str = str(cur_frameId) + sep + str(activity) + content_str + '\n'
		write_txt(new_file, content_str, 'a')
		num +=1
		return num

	def __getPersons(video_folder, save_root):
		tracker = dlib.correlation_tracker()
		win = dlib.image_window()
		ranked = True
		for root, dirs, files in os.walk(data_folder):
			for file in files:

				# modifying...
				
				if file.split('.')[0] == 'annotations':
					video_id = int(root.split('/')[-1].split('seq')[1])
					print video_id
					num_clips = __modify_annotation(os.path.join(root, file))
					if video_id in trainval_videos:
						total_train += num_clips
					else:
						total_test += num_clips
					print total_train, total_test
				

				# tracking by Dlib...
				if file.split('.')[0] == 'annotations_new':
					video_id = int(root.split('/')[-1].split('seq')[1])
					img_list = sorted(glob.glob(os.path.join(root, "*.jpg")))
					print video_id, len(img_list)
					lines = open(os.path.join(root, file)).readlines()
					for line in lines:
						frame_id, rects = __modify_annotation(line)
						frame_id = int(frame_id)
						imgs={}
						if frame_id + 4 <=len(img_list) and frame_id-5>0:
							'''
							labeled_frame_name = os.path.join(root, 'frame%04d.jpg'%frame_id)
							if not os.path.exists(labeled_frame_name):
								print 'Labeled frame doesn\'t exists!'
								break
							'''
							clip_list = img_list[frame_id-6:frame_id+4]
							imgs['pre'] = clip_list[:5][::-1]
							imgs['back'] = clip_list[4:]
							#print imgs
							save_path = os.path.join(save_root, str(video_id), str(frame_id))
							if not(os.path.exists(save_path)):
								os.makedirs(save_path)
								if ranked == True:
									# do rank
									pass
								# We will track the frames as we load them off of disk
								track(rects, imgs, tracker, win, save_path)


	def __getTrainTest(img_folder):
		dataset_config={'trainval':self.trainval_videos,'test':self.test_videos}
		dataList={'trainval':[],'test':[]}
		activity_label_list = {'trainval':[],'test':[]}

		print 'trainval_videos:', dataset_config['trainval']
		print 'test_videos:', dataset_config['test']

		for phase in ['trainval','test']:
			action_list = []
			activity_list = []
			Count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			for idx in dataset_config[phase]:
				video_path = img_folder  + str(idx) + '/'
				for root, dirs, files in os.walk(video_path):
					activity_label = ''
					if len(files)!=0:
						if len(files)/10>8:
							print root, len(files)/10
						Count[len(files)/10-1] += 1
						
						files.sort()
						for i in xrange(self.K_players*T):
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
			
			print Count