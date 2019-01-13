"""
	Group_Activity Test_Demo
"""
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import datasets, svm, metrics
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append('../')
import models
import utils
import numpy as np
import Solver
from dataset import *
import matplotlib.pyplot as plt

class_names = ['lpass', 'rpass', 'lset', 'rset', 'lspike', 'rspike', 'lwin', 'rwin']


K_players = 12
T = 10
use_gpu = torch.cuda.is_available()

def label_map(labels):
	Map = [0,5,1,2,3,4,6,7]
	for i in range(labels.shape[0]):
		labels[i] = Map[labels[i]]
	return labels

def normlize(matrix):
	matrix = np.asarray(matrix, dtype=float)
	return np.round(matrix/np.sum(matrix, 1).reshape(-1,1)*100, decimals=2)

def getPerson_level():
	"""
		Given a video, output CNNLSTM_type faeture
	"""
	
	# dataset
	dataset_Class = 'VolleyballDataset_img'
	#dataset_Class = 'CADataset_img'
	# dataset
	if dataset_Class == 'VolleyballDataset_img':
		Num_classes = 9
		#data_root = '/home/ubuntu/caffe-lstm/examples/deep-activity-rec/eclipse-project/fused_data/'
		data_root = '/home/ubuntu/GCNN/dataset/VD/person_imgs/'
		K_players = 12
	else:
		Num_classes = 5
		data_root = '/home/ubuntu/GCNN/dataset/CAD/person_imgs/'
		K_players = 5

	#data_root = '/home/ubuntu/caffe-lstm/examples/deep-activity-rec/eclipse-project/fused_data/'
	#data_root = '/home/ubuntu/GCNN/dataset/volleyball_imgs/'
	#image_datasets = {x: VolleyballDataset_img(data_root, x, 'activity') for x in ['trainval', 'test']}
	image_datasets = {x: eval(dataset_Class)(data_root, x, 'activity') for x in ['trainval', 'test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=K_players*10, shuffle=False, num_workers=4) for x in ['trainval', 'test']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['trainval', 'test']}
	save_prefix = 'VD_CNNLSTM_features_'

	cnn_lstm = models.alexNet_LSTM(pretrained=False, num_classes=Num_classes)
	cnn_lstm.load_state_dict(torch.load('/home/ubuntu/MM/Run/VD_CNNLSTM.pkl'))
	if use_gpu:
		cnn_lstm = cnn_lstm.cuda()

	utils.extract_features(cnn_lstm, dataloaders, dataset_sizes, T, K_players, 7096*K_players, save_prefix)


def getGroup_level():
	"""
	"""

	# Group_level
    # init
	num_epochs = 100
	batch_size = {'trainval':500,'test':10}# 500, 10
	step_size = 10
	gamma = 0.1
	# data
	train_dataFile = '/home/ubuntu/MM/CNNLSTM_features_trainval.npy'
	test_dataFile = '/home/ubuntu/MM/CNNLSTM_features_test.npy'
	# model
	#net = models.Group_level.bilstm_Maxpooling(input_size = 7096, hidden_size = 1000, K_players = K_players)
	net = models.one_to_all(pretrained=True, input_size = 7096, hidden_size = 1000, K_players = K_players)
	if use_gpu:
		net = net.cuda()

	# define a loss function and optimizer
	criterion = torch.nn.CrossEntropyLoss()  # softmax + nll_loss
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.9))

	solver = Solver.Solver(net, criterion, optimizer, step_size = step_size, gamma = gamma, num_epochs = num_epochs, batch_size = batch_size, use_gpu = use_gpu)
	solver.set_Data(dataFile = {'trainval':train_dataFile, 'test': test_dataFile})
	solver.extract_features()





def Group_level():
	"""
		Given a video, output a group activity label
	"""

	# data: read 120 frames data of a video from *.npy
	testdata_File = '/home/ubuntu/MM/CNNLSTM_features_test.npy'
	testdata = np.load(testdata_File, mmap_mode = 'r')

	batchsize = 10

	# model: bilstm_maxpooling
	net = models.one_to_all(pretrained=True, input_size=7096, hidden_size=1000, K_players=K_players)
	net = net.cuda()
	net.train(False)
	preds = []
	labels = []
	# forward
	for i in range(len(testdata)/batchsize):
		#print i*batchsize, batchsize*(i+1)
		data = testdata[i*batchsize:batchsize*(i+1)]
		inputs = torch.from_numpy(data[:,:-1]).float()
		label = data[0,-1]
		if use_gpu:
			inputs = inputs.cuda()
		inputs = Variable(inputs, requires_grad=False)
		#print inputs.size()
		_, outputs = net.forward(inputs)
		probs = F.softmax(outputs.data)
		#print torch.sum(probs,1)
		#print probs
		_, pred = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
		preds.append(pred.cpu().numpy())
		labels.append(label)
		#break
	### step 5: result
	preds = np.asarray(preds, dtype=int)
	#preds = label_map(preds)
	labels = np.asarray(labels, dtype=int)
	#labels = label_map(labels)
	preds, labels = preds.reshape(-1,1), labels.reshape(-1,1)
	print("Classification report for classifier \n %s" % (metrics.classification_report(labels, preds)))
	print("Confusion matrix:\n%s" % normlize(metrics.confusion_matrix(labels, preds)))
	print np.sum(preds == labels), preds.shape, labels.shape
	print np.sum(preds == labels) / float(labels.shape[0])


	# Compute confusion matrix
	cnf_matrix = metrics.confusion_matrix(labels, preds)
	print cnf_matrix
	np.set_printoptions(precision=2)

	plt.rc('text',usetex=True)
	plt.rc('font', family='serif')
	# Plot non-normalized confusion matrix
	plt.figure()
	utils.plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='')

	# Plot normalized confusion matrix
	plt.figure()
	utils.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='')

	plt.show()

def Scene_level():
	"""
		Given a video, output a group activity label
	"""

	# data: read 120 frames data of a video from *.npy
	testdata_File = '/home/ubuntu/MM/Group_features_test.npy'
	testdata = np.load(testdata_File, mmap_mode = 'r')

	batchsize = 10

	# model: bilstm_maxpooling
	net = models.scene_lstm(pretrained=True, input_size=2000, hidden_size=1000, K_players=K_players)
	net = net.cuda()
	net.train(False)
	preds = []
	labels = []
	# forward
	for i in range(len(testdata)/batchsize):
		#print i*batchsize, batchsize*(i+1)
		data = testdata[i*batchsize:batchsize*(i+1)]
		inputs = torch.from_numpy(data[:,:-1]).float()
		label = data[0,-1]
		if use_gpu:
			inputs = inputs.cuda()
		inputs = Variable(inputs, requires_grad=False)
		#print inputs.size()
		_, outputs = net.forward(inputs)
		probs = F.softmax(outputs.data)
		#print torch.sum(probs,1)
		#print probs
		_, pred = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
		preds.append(pred.cpu().numpy())
		labels.append(label)
		#break
	### step 5: result
	preds = np.asarray(preds, dtype=int)
	preds = label_map(preds)
	labels = np.asarray(labels, dtype=int)
	labels = label_map(labels)
	preds, labels = preds.reshape(-1,1), labels.reshape(-1,1)
	print("Classification report for classifier \n %s" % (metrics.classification_report(labels, preds)))
	print("Confusion matrix:\n%s" % normlize(metrics.confusion_matrix(labels, preds)))
	print np.sum(preds == labels), preds.shape, labels.shape
	print np.sum(preds == labels) / float(labels.shape[0])

	# Compute confusion matrix
	cnf_matrix = metrics.confusion_matrix(labels, preds)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	utils.plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plt.figure()
	utils.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

	plt.show()

if __name__ == '__main__':
	getPerson_level()
	#getGroup_level()
	#Group_level()
	#Scene_level()