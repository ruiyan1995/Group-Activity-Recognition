# baseline1: classification by personal action class statistics
import os
import numpy as np
from collections import deque
from sklearn import datasets, svm, metrics, preprocessing
from skimage import io, transform
#import models
import torch
from torch.autograd import Variable

Total_Num = 0
Num = 0
K_players = 12
Personal_Class_Num = 9



Person_labelName = {'blocking': 4, 'digging': 2, 'falling': 5, 'jumping': 8,
	'moving': 1, 'setting': 6, 'spiking': 3, 'standing': 0, 'waiting': 7}
#Person_labelName=['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
Frame_labelName =['l-pass', 'r-pass', 'l_set', 'r_set', 'l-spike', 'r_spike', 'l_winpoint', 'r_winpoint']
frame_ID_Dict = dict()
frame_Label_Dict = dict()


def getPersonal_label_Matrix(annotation_matrix):
	# get personal action class label sorted by X
	# inputs: annotation Rects with labels [x, y, w, h, label] [N*K_players*5]
	#print annotation_matrix
	N = annotation_matrix.shape[0]
	Personal_label_Matrix = np.zeros([N, K_players])
	for i in range(N):
		indx = np.argsort(annotation_matrix[i, :, 0])
		sorted_labels = annotation_matrix[i, :, -1][indx]
		Personal_label_Matrix[i] = sorted_labels

	return Personal_label_Matrix

def get_word_vectors(label_matrix, description_type = 'Bow'):
	# inputs: label matrix [N*6]
	# outputs: two concated Bag of words type vector [N*8]
	if np.max(label_matrix)!=(Personal_Class_Num-1):
		print 'error............'
	else:
		N = label_matrix.shape[0]
		feature = np.zeros([N, Personal_Class_Num])
		for i in range(N):
			for j in range(label_matrix.shape[1]):
				t = int(label_matrix[i][j])
				if description_type == 'Bow':
					# Bag of word
					feature[i][t] = feature[i][t] + 1
				elif description_type == 'One_hot':
					# one hot
					feature[i][t] = 1
	return feature



def getFeature(Personal_label_Matrix, group_num = 2, description_type = 'Bow'):
	"""
	"""
	print Personal_label_Matrix.shape
	w_group = K_players / group_num
	N = Personal_label_Matrix.shape[0]
	Group_feature = np.zeros([group_num, N, Personal_Class_Num])
	for g in range(group_num):
		Group_feature[g] = get_word_vectors(Personal_label_Matrix[:, w_group * g : w_group * (g+1)], description_type = description_type)
	print Group_feature[0].shape,Group_feature[1].shape
	##### need to improve
	feature = np.concatenate((Group_feature[0], Group_feature[1]), axis = 1)
	print feature.shape
	return feature


def getTrain_Test_Splits(dataset_config_floader):
	""" read txt in dataset_config_floader, 
	['trainval', 'test']

	"""
	video_id_list = {x: [] for x in ['trainval', 'test']}
	for phase in ['trainval', 'test']:
		file = open(dataset_config_floader + phase + '.txt')
		lines = file.readlines()
		for line in lines:
			video_id_list[phase].append(line.split('\n')[0])
	video_id_array = {x: np.array(video_id_list[x]) for x in ['trainval', 'test']}
	#print video_id_array['test']
	return video_id_array

def Annotation_StringParse(annotation, rects_with_label):
	"""String Parse
	
	Args:
		annotation (str): frame_id, activity, [x,y,w,h,action], ...
		rects_with_label (bool): 

	Returns:
		frame_id, frame_label, Rects

	""" 
	Rects = np.zeros([K_players, 5], dtype=int)# note: the raw_num of rects will be less than K_players
	keywords = deque(annotation.strip().split(' '))
	frame_id = keywords.popleft().split('.')[0]
	activity = Frame_labelName.index(keywords.popleft())
	i = 0
	
	while keywords:
		x = int(keywords.popleft())
		y = int(keywords.popleft())
		w = int(keywords.popleft())
		h = int(keywords.popleft())
		action = Person_labelName[keywords.popleft()]
		if rects_with_label:
			Rects[i] = [x,y,w,h,action]
		else:
			Rects[i] = [x,y,w,h,-1]
		
		i = i + 1

	return frame_id, activity, Rects

def predict(img_path, Rects, net):
	"""
	Args:
		frame_id:
		Rects:
	Return:
		New_Rects:
	"""
	# get a batch of inputs, note: batch = K_players * 10 = 120
	imgs = np.zeros([K_players,3,224,224])
	preds = np.zeros([K_players, 1],dtype=int)
	#print img_path
	img = io.imread(img_path)
	for i in range(len(Rects)):
		x,y,w,h = Rects[i][:-1]
		if w==0 and h==0:
			break
		else:
			imgs[i]=transform.resize(img[y:y+h,x:x+w],(224,224),mode='constant').transpose((2, 0, 1))
	
	# put imgs into CNN_model
	imgs = imgs[:i]
	imgs = imgs.repeat(10, axis=0)
	#print imgs.shape
	inputs = torch.from_numpy(imgs).float()
	#print inputs.size()
	inputs = Variable(inputs.cuda(), requires_grad=False)
	outputs = net.forward(inputs)
	#print outputs.data.size()
	#print torch.mean(outputs.data, 0).size()
	for j in range(i):
		_, pred = torch.max(torch.mean(outputs.data[10*j:10*(j+1)], 0).view(1,-1),1)
		#print pred.cpu().numpy()
		preds[j] = pred.cpu().numpy()
	#_, preds = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
	preds = preds[:i]
	#global Num, Total_Num
	#Num = Num + np.sum(Rects[:i,-1] == preds.reshape(-1))
	#Total_Num = Total_Num + len(Rects)
	#print Num/float(Total_Num)
	Rects[:i,-1] = preds.reshape(-1)

	return Rects



def Annotation_matrixs_Generation(dataset_floader, train_test_splits):
	"""To Generate Annotation_matrixs
	Args:
		dataset_floader (str):
		train_test_splits (dict):
	Returns:
		Matrix:
	"""
	print 'Generating.........'
	N = {'trainval': 3493,'test':1337}
	exts = ["txt"]
	for phase in ['trainval', 'test']:
		if phase == 'test':
			net = models.alexNet_LSTM(pretrained=True, num_classes=9)
			net = net.cuda()
			net.load_state_dict(torch.load('CNNLSTM.pkl'))

		video_id = train_test_splits[phase]
		RectsArray = np.zeros([N[phase], K_players, 5])
		FrameLabelArray = np.zeros([N[phase], 1])
		i=0
		for subdir, dirs, files in os.walk(dataset_floader):
			for fileName in files:
				if any(fileName.lower().endswith("." + ext) for ext in exts):
					#print fileName
					current_video_id = subdir.split('/')[-1]
					if current_video_id in video_id:
						#print subdir.split('/')[-1]
						annotations = open(os.path.join(subdir, fileName))
						for line in annotations:
							frame_id, frame_label, Rects = Annotation_StringParse(line, rects_with_label = True)
							
							#frame_ID_Dict[i] = frame_id
							#frame_Label_Dict[frame_id]=frame_label
							FrameLabelArray[i] = frame_label
							if phase == 'test':
								img_path = dataset_floader + current_video_id + '/' + frame_id + '/' + frame_id + '.jpg'
								#print np.array_equal(Rects, predict(img_path, Rects, net))
								Rects = predict(img_path, Rects, net)
							RectsArray[i] = Rects 
							i = i + 1
							#print frame_id, ' ', frame_label, Rects
		np.savez(dataset_floader + phase + '_Annotation_matrix', Rects = RectsArray, Frame_label = FrameLabelArray)




if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	### step 1: product annotation file for all video
	project_floader = '/home/ubuntu/GAR/dataset/'
	dataset_floader = os.path.join(project_floader, 'VD/feas/')
	dataset_config_floader = os.path.join(project_floader, 'imgs')
	fea_save_prefix = '_feas_with_label_'
	feature_type = 'One_hot'
	feature_type = 'Bow'
	feature_type = 'TF_IDF'
	'''
	# check for existence............
	print os.path.join(dataset_floader, 'trainval_Annotation_matrix_GT.npz')
	if os.path.exists(os.path.join(dataset_floader, 'trainval_Annotation_matrix_GT.npz')) == False:
		Annotation_matrixs_Generation(dataset_floader, getTrain_Test_Splits(dataset_config_floader))
	else:
		print 'Annotation_matrixs existing...'
	
	### step 2: extract feature......
	if os.path.exists(os.path.join(dataset_floader + feature_type + fea_save_prefix + 'trainval' + '.npy')) == False:
		### step 2.1: load Annotation matrix(**.npy).........
		Annotation_matrixs = {x: np.load(os.path.join(dataset_floader, x + '_Annotation_matrix_GT.npz'))['Rects'] for x in ['trainval', 'test']}
		Frame_label = {x: np.load(os.path.join(dataset_floader, x + '_Annotation_matrix_GT.npz'))['Frame_label'] for x in ['trainval', 'test']}
		
		### step 2.2: generating features.......
		print 'Generating ' + feature_type + ' features.........'
		for phase in ['trainval', 'test']:
			Personal_label_Matrix = getPersonal_label_Matrix(Annotation_matrixs[phase])
			frame_fea = getFeature(Personal_label_Matrix, description_type = feature_type)
			frame_label = Frame_label[phase]
			data = np.concatenate((frame_fea, frame_label), axis = 1)
			np.save(feature_type + fea_save_prefix + phase + '.npy', data)
	else:
		print 'Features existing...'
	'''
	### step 3: trainning........
	data = np.load(os.path.join(dataset_floader + feature_type + '_trainval' + '.npy'))
	train_data = data[:,:-1]
	train_label = data[:,-1]
	train_data = preprocessing.normalize(train_data, norm='l2')
	print train_data[100:120]
	print train_label[100:120]
	classifier = svm.SVC(gamma = 0.01)
	classifier.fit(train_data, train_label)

	### step 4: testing..........
	data = np.load(os.path.join(dataset_floader + feature_type + '_test' + '.npy'))
	test_data = data[:,:-1]
	expected = data[:,-1]
	test_data = test_data / np.linalg.norm(test_data)
	test_data = preprocessing.normalize(test_data, norm='l2')
	predicted = classifier.predict(test_data)
	

	### step 5: result
	print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	