import numpy as np
import os
import glob
import dlib
from collections import deque
from skimage import io, transform
import utils
Class_Num = 5

trainval_videos = [7,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
test_videos = [1,2,3,4,5,6,8,9,10,11,25,28,29]


#test_videos = [1,4,5,6,8,2,7,28,35,11,10,26]
#test_videos = [9,15,19,20,3,16,36,38,12,40,41]
#test_videos = [21,22,23,24,17,43,37,13,25,39]
#test_videos = [32,33,42,44,18,31,29,34,14,27]
trainval_videos = list(set(range(1,45)).difference(set(test_videos)))
total_train = 0
total_test = 0


def track(person_rects, imgs, tracker, win, save_path):
	for i, person in enumerate(person_rects):
		for j, phase in enumerate(['pre', 'back']):
			if j == 0:
				j = -1
			for k, f in enumerate(imgs[phase]):
				#print("Processing Frame {}".format(k))
				img = io.imread(f)
				if k == 0:
					x, y, w, h, action, activity = person
					#print x,y,w,h
					tracker.start_track(img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
				else:
					tracker.update(img)

				pos = tracker.get_position()
				top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
				cropped_image = img[top:bottom,left:right]
				cropped_image = transform.resize(np.ascontiguousarray(cropped_image),(256,256),mode='constant')
				img_name = os.path.join(save_path, "%04d_%d_%d.jpg"%(10*i+(5+j*k), action, activity))
				#print img_name
				io.imsave(img_name, cropped_image)


def old_track(person_rects, imgs, tracker, win, save_path):
	for i, person in enumerate(person_rects):
		for k, f in enumerate(imgs):
			#print("Processing Frame {}".format(k))
			img = io.imread(f)
			if k == 0:
				x, y, w, h, action, activity = person
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
			img_name = os.path.join(save_path, "%04d_%d_%d.jpg"%(10*i+k+1, action, activity))
			#print img_name
			io.imsave(img_name, cropped_image)

	#print save_path, len(person_rects)


def parse_CAD_Annotations(line):
	keywords = deque(line.strip().split(' '))
	frame_id = keywords.popleft().split('.')[0]
	activity = int(keywords.popleft())
	#Rects = np.zeros([K_players, 4])
	
	Rects = []
	i=0
	while keywords:
		x = int(keywords.popleft())
		y = int(keywords.popleft())
		w = int(keywords.popleft())
		h = int(keywords.popleft())
		#Rects[i] = [x,y,w,h]
		action = int(keywords.popleft())
		Rects.append([x,y,w,h,action,activity])
		i+=1
		#keywords.popleft()
	Rects = np.asarray(Rects)
	# sort Rects by the first col
	Rects = Rects[np.lexsort(Rects[:,::-1].T)]
	'''
	Points = np.zeros([len(Rects), 2])
	for i, rect in enumerate(Rects):
		x, y, w, h, _, _ = rect
		Points[i, 0], Points[i, 1] = x + w/2, y + h/2
	'''
	#return frame_id, Points, Rects
	return frame_id, Rects
	
def write_txt(txtFile, content_str, mode):
	with open(txtFile, mode) as f:
		f.write(content_str)

def modify_annotation(annotation_file):

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
				write_txt(new_file, content_str, 'a')
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


def generate_train_test(img_folder):
	"""
		Args:
			img_folder: 
		Return:
	"""
	K_players = 5
	Total_videos = 44
	T = 10
	
	dataset_config={'trainval':trainval_videos,'test':test_videos}
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
					for i in xrange(K_players*T):
						if i<len(files):
							# parse
							filename = files[i]
							action_label = filename.split('_')[1]
							activity_label = filename.split('_')[2].split('.')[0]
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
		action_list = utils.block_shuffle(action_list, T)
		txtFile = img_folder + phase + '_action' + '.txt'
		open(txtFile, 'w')
		print phase +'_size:' + str(len(action_list)/(T*K_players))
		for i in range(len(action_list)):
			with open(txtFile, 'a') as f:
				f.write(action_list[i])
		
		# write activity txt
		activity_list = utils.block_shuffle(activity_list, T*K_players)
		txtFile = img_folder + phase + '_activity' + '.txt'
		open(txtFile, 'w')
		print phase +'_size:' + str(len(activity_list)/(T*K_players))
		for i in range(len(activity_list)):
			with open(txtFile, 'a') as f:
				f.write(activity_list[i])
		
		print Count

def generate_data(video_folder, save_root):
	tracker = dlib.correlation_tracker()
	win = dlib.image_window()
	ranked = True
	for root, dirs, files in os.walk(data_folder):
		for file in files:

			# modifying...
			'''
			if file.split('.')[0] == 'annotations':
				video_id = int(root.split('/')[-1].split('seq')[1])
				print video_id
				num_clips = modify_annotation(os.path.join(root, file))
				if video_id in trainval_videos:
					total_train += num_clips
				else:
					total_test += num_clips
				print total_train, total_test
			'''

			# tracking by Dlib...
			if file.split('.')[0] == 'annotations_new':
				video_id = int(root.split('/')[-1].split('seq')[1])
				img_list = sorted(glob.glob(os.path.join(root, "*.jpg")))
				print video_id, len(img_list)
				lines = open(os.path.join(root, file)).readlines()
				for line in lines:
					frame_id, rects = parse_CAD_Annotations(line)
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

if __name__ == '__main__':

	data_folder = '/home/ubuntu/CAD/CAD1/ActivityDataset'
	save_root = '/home/ubuntu/CAD/CAD1/Dlib_imgs_ranked/'

	#generate_data(data_folder, save_root)
	generate_train_test(img_folder=save_root)

