import os
import numpy as np
import math
from collections import deque
from sklearn import datasets, svm, metrics
from skimage import io, transform
import torch
from torch.autograd import Variable
from collections import Counter
import Models


dataset_root = '/home/ubuntu/GAR/dataset/VD/imgs'

np.set_printoptions(suppress=True)


class VD_Semantic(object):
    """docstring for VD_Semantic"""

    def __init__(self, dataset_root, dataset_confs, model_confs):
        super(VD_Semantic, self).__init__()
        self.INF = 99999
        self.num_players = dataset_confs.num_players
        self.num_classes = dataset_confs.num_classes
        self.action_list = dataset_confs.action_list
        self.activity_list = dataset_confs.activity_list
        self.splits = dataset_confs.splits
        self.dataset_floader = os.path.join(dataset_root, 'VD/imgs/')
        #self.feature_type = 'One_hot'
        #self.feature_type = 'Bow'
        self.feature_type = 'TF_IDF'
        self.fea_save_prefix = '_feas_with_label_'
        self.GT = False
        self.suffix = 'GT' if self.GT else 'Pred'

        self.model_confs = model_confs

        # step 1: product annotation file for all video
        # check for existence............
        print os.path.join(self.dataset_floader, 'trainval_Annotation_matrix.npz')
        if os.path.exists(os.path.join(self.dataset_floader, 'trainval_Annotation_matrix_' + self.suffix + '.npz')) == False:
            self.Annotation_matrixs_Generation(
                self.dataset_floader, self.splits, GT=self.GT)
        else:
            print 'Annotation_matrixs existing...'

        # step 2: extract feature......
        if os.path.exists(os.path.join(self.dataset_floader + self.feature_type + self.fea_save_prefix + 'trainval_' + self.suffix + '.npy')) == False:
            # step 2.1: load Annotation matrix(**.npy).........
            print os.path.join(
                self.dataset_floader, 'trainval' + '_Annotation_matrix_' + self.suffix + '.npz')
            Annotation_matrixs = {x: np.load(os.path.join(
                self.dataset_floader, x + '_Annotation_matrix_' + self.suffix + '.npz'))['Rects'] for x in ['trainval', 'test']}
            Frame_label = {x: np.load(os.path.join(
                self.dataset_floader, x + '_Annotation_matrix_' + self.suffix + '.npz'))['Frame_label'] for x in ['trainval', 'test']}
            # step 2.2: generating features.......
            print 'Generating ' + self.feature_type + ' features.........'
            for phase in ['trainval', 'test']:
                Personal_label_Matrix = self.getPersonal_label_Matrix(
                    Annotation_matrixs[phase])
                frame_fea = self.getFeature(
                    Personal_label_Matrix, description_type=self.feature_type)
                frame_label = Frame_label[phase]
                data = np.concatenate((frame_fea, frame_label), axis=1)
                np.save(os.path.join(self.dataset_floader, self.feature_type +
                                     self.fea_save_prefix + phase + '_' + self.suffix + '.npy'), data)
        else:
            print 'Features existing...'

    def Annotation_matrixs_Generation(self, dataset_floader, trainval_test_splits, GT=False):
        """To Generate Annotation_matrixs
        Args:
                dataset_floader (str):
                train_test_splits (dict):
                GT(bool): if True, the labels of trainval and test are all groud truth, 
                            else the labels of test phase are predicted.
        Returns:
                Matrix:
        """
        print 'Generating.........'
        N = {'trainval': 3493, 'test': 1337}
        exts = ["txt"]
        for phase in ['trainval', 'test']:
            if GT == False and phase == 'test':
                net = Models.resnet50_LSTM(
                    pretrained=True, model_confs=self.model_confs)
                #net = models.alexNet_LSTM(pretrained=True, num_classes=9)
                net = net.cuda()
                net.load_state_dict(torch.load(
                    'weights/VD/action/resnet50_LSTM.pkl'))

            video_id = trainval_test_splits[phase]
            RectsArray = np.zeros([N[phase], self.num_players, 5])
            FrameLabelArray = np.zeros([N[phase], 1])
            i = 0
            for subdir, dirs, files in os.walk(dataset_floader):
                for fileName in files:
                    if any(fileName.lower().endswith("." + ext) for ext in exts):
                        # print fileName
                        current_video_id = subdir.split('/')[-1]
                        if current_video_id != '' and int(current_video_id) in video_id:
                            # print subdir.split('/')[-1]
                            annotations = open(os.path.join(subdir, fileName))
                            for line in annotations:
                                frame_id, frame_label, Rects = self.AnnotationString_Parse(
                                    line, rects_with_label=True)
                                #frame_ID_Dict[i] = frame_id
                                # frame_Label_Dict[frame_id]=frame_label
                                FrameLabelArray[i] = frame_label
                                if GT == False and phase == 'test':
                                    # just predict the label of one frame???
                                    img_path = os.path.join(dataset_floader, current_video_id, frame_id, frame_id + '.jpg')
                                    # print np.array_equal(Rects,
                                    # predict(img_path, Rects, net))
                                    Rects = self.predict(img_path, Rects, net)

                                #RectsList.append(Rects)
                                Rects = checkRects(Rects)
                                RectsArray[i] = Rects
                                i = i + 1
                                
                                print frame_id, ' ', frame_label, Rects
            np.savez(dataset_floader + phase + '_Annotation_matrix_' + self.suffix,
                     Rects=RectsArray, Frame_label=FrameLabelArray)

    def getPersonal_label_Matrix(self, annotation_matrix):
        # get personal action class label sorted by X
        # Since the labels of each bounding boxes are unordered in annotation file.
        # inputs: annotation Rects with labels [x, y, w, h, label]
        # [N*K_players*5]
        
        N = annotation_matrix.shape[0]
        Personal_label_Matrix = np.zeros([N, self.num_players])
        for i in range(N):
            indx = np.argsort(annotation_matrix[i, :, 0])
            sorted_labels = annotation_matrix[i, :, -1][indx]
            Personal_label_Matrix[i] = sorted_labels

        return Personal_label_Matrix

    '''
    def get_word_vectors(self, label_matrix, description_type='Bow'):
        # inputs: label matrix [N*6]
        # outputs: two concated Bag of words type vector [N*8]
        if np.max(label_matrix) != (self.num_classes['action'] - 1):
            print 'error............'
        else:
            N = label_matrix.shape[0]
            feature = np.zeros([N, self.num_classes['action']])
            for i in range(N):

                for j in range(label_matrix.shape[1]):
                    t = int(label_matrix[i][j])
                    if t == -1:
                        print 'pass!'
                        pass
                    else:
                        if description_type == 'Bow':
                            # Bag of word
                            #feature[i][t] = feature[i][t] + 1
                            feature[i][t] = count[t]
                        elif description_type == 'One_hot':
                            # one hot
                            feature[i][t] = 1
                        elif description_type == 'TF_IDF':
                            feature[i][t] = TF_IDF(
                                label_matrix[i], count, count_list)
        return feature'''

    def get_word_vectors(self, corpus, description_type='TF_IDF'):
        count_list = []
        feature = np.zeros([len(corpus), self.num_classes['action']])

        for doc in corpus:
            count_list.append(Counter(doc))

        for d, doc in enumerate(corpus):
            for word in doc:
                if word == -1:
                    print 'pass'
                    pass
                else:
                    if description_type == 'TF_IDF':
                        #print word, count_list[d], count_list
                        feature[d][word] = self.TF_IDF(
                            word, count_list[d], count_list)
                    elif description_type == 'Bow':
                        feature[d][word] = count_list[d][word]
                    elif description_type == 'One_hot':
                        feature[d][word] = 1
        print feature
        exit(0)
        return feature

    def n_containing(self, word, count_list):
        return sum(1 for count in count_list if word in count)

    def TF_IDF(self, word, count, count_list):
        tf = float(count[word]) / sum(count.values())
        idf = math.log(len(count_list)) / \
            (1 + self.n_containing(word, count_list))
        return tf * idf

    def getFeature(self, Personal_label_Matrix, group_num=2, description_type='Bow'):
        print Personal_label_Matrix.shape
        w_group = self.num_players / group_num
        N = Personal_label_Matrix.shape[0]
        Group_feature = np.zeros([group_num, N, self.num_classes['action']])
        for g in range(group_num):
            Group_feature[g] = self.get_word_vectors(Personal_label_Matrix[
                :, w_group * g: w_group * (g + 1)], description_type=description_type)
        print Group_feature[0].shape, Group_feature[1].shape
        # need to improve
        feature = np.concatenate((Group_feature[0], Group_feature[1]), axis=1)
        print feature.shape
        return feature

    def checkRects(self, Rects):
        # note: the raw_num of rects will be less than K_players, so we need reset the bad record such as [0,0,0,0,0]
        for i, rect in enumerate(Rects):
            if rect == [0,0,0,0,0]:
                Rects[i] = [self.INF,0,0,0,-1]
        return Rects

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


    def AnnotationString_Parse(self, annotation_str, rects_with_label):
        """String Parse
        Args:
                annotation (str): frame_id, activity, [x,y,w,h,action], ...
                rects_with_label (bool): 
        """
        Rects = np.zeros(
            [self.num_players, 5], dtype=int)  # note: the raw_num of rects will be less than K_players
        keywords = deque(annotation_str.strip().split(' '))
        frame_id = keywords.popleft().split('.')[0]
        activity = self.activity_list.index(keywords.popleft())
        i = 0
        while keywords:
            x = int(keywords.popleft())
            y = int(keywords.popleft())
            w = int(keywords.popleft())
            h = int(keywords.popleft())
            #action = Person_labelName[keywords.popleft()]
            action = int(self.action_list.index(keywords.popleft()))
            if rects_with_label:
                Rects[i] = [x, y, w, h, action]
            else:
                Rects[i] = [x, y, w, h, -1]
            i = i + 1

        # sort Rects by the first col
        Rects = Rects[np.lexsort(Rects[:,::-1].T)]
        return frame_id, activity, Rects


    def AnnotationString_Parse(self, line):
        keywords = deque(line.strip().split(' '))
        frame_id = keywords.popleft().split('.')[0]
        activity = self.activityList.index(keywords.popleft())
        Rects = []
        while keywords:
            x = int(keywords.popleft())
            y = int(keywords.popleft())
            w = int(keywords.popleft())
            h = int(keywords.popleft())
            action = int(self.actionList.index(keywords.popleft()))
            Rects.append([x,y,w,h,action])
        Rects = np.asarray(Rects)
        # sort Rects by the first col
        Rects = Rects[np.lexsort(Rects[:,::-1].T)]
        return frame_id, activity, Rects

    def predict(self, img_path, Rects, net):
        """
        Args:
            frame_id:
            Rects:
        Return:
            New_Rects:
        """
        # get a batch of inputs, note: batch = K_players * 10 = 120
        imgs = np.zeros([self.num_players, 3, 224, 224])
        preds = np.zeros([self.num_players, 1], dtype=int)
        # print img_path
        img = io.imread(img_path)
        for i in range(len(Rects)):
            x, y, w, h = Rects[i][:-1]
            if w == 0 and h == 0:
                break
            else:
                imgs[i] = transform.resize(
                    img[y:y + h, x:x + w], (224, 224), mode='constant').transpose((2, 0, 1))

        # put imgs into CNN_model
        imgs = imgs[:i]
        imgs = imgs.repeat(10, axis=0)
        # print imgs.shape
        inputs = torch.from_numpy(imgs).float()
        # print inputs.size()
        inputs = Variable(inputs.cuda(), requires_grad=False)
        _, outputs = net.forward(inputs)
        # print outputs.data.size()
        # print torch.mean(outputs.data, 0).size()
        for j in range(i):
            _, pred = torch.max(torch.mean(
                outputs.data[10 * j:10 * (j + 1)], 0).view(1, -1), 1)
            # print pred.cpu().numpy()
            preds[j] = pred.cpu().numpy()
        #_, preds = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
        preds = preds[:i]
        #global Num, Total_Num
        #Num = Num + np.sum(Rects[:i,-1] == preds.reshape(-1))
        #Total_Num = Total_Num + len(Rects)
        # print Num/float(Total_Num)
        Rects[:i, -1] = preds.reshape(-1)

        return Rects
