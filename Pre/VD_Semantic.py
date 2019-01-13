import os
import glob
import numpy as np
import math
import dlib
from collections import deque
from sklearn import datasets, svm, metrics
from skimage import io, transform
import torch
from torch.autograd import Variable
from collections import Counter
import Models
import utils


dataset_root = '/home/ubuntu/GAR/dataset/VD/imgs'

np.set_printoptions(threshold=np.inf)


class VD_Semantic(object):
    """docstring for VD_Semantic"""

    def __init__(self, dataset_root, dataset_confs, model_confs):
        super(VD_Semantic, self).__init__()
        self.INF = 99999
        self.tracker = dlib.correlation_tracker()
        self.num_players = dataset_confs.num_players
        self.num_classes = dataset_confs.num_classes
        self.action_list = dataset_confs.action_list
        self.activity_list = dataset_confs.activity_list
        self.splits = dataset_confs.splits
        self.dataset_floader = os.path.join(dataset_root, 'VD/imgs/')
        self.save_floader = os.path.join(dataset_root, 'VD/feas/')
        #self.feature_type = 'One_hot'
        #self.feature_type = 'Bow'
        self.feature_type = 'TF_IDF'
        self.GT = False
        self.model_confs = model_confs
        self.Pred_file_path = os.path.join(self.dataset_floader, 'Preds.txt')
        if os.path.exists(self.Pred_file_path):
            self.preds_Dicts = self.get_Preds(self.Pred_file_path)
        else:
            self.preds_Dicts = None
        # step 1: Parse annotation file for all video
        RectsList, FrameLabelList = self.Annotation_matrixs_Generation(
            self.dataset_floader, self.splits, GT=self.GT)

        # step 2: Extract feature......
        if os.path.exists(os.path.join(self.save_floader + self.feature_type + 'trainval' + '.npy')) == False:
            # step 2.2: generating features.......
            print 'Generating ' + self.feature_type + ' features.........'
            for phase in ['trainval', 'test']:
                corpus = self.getCorpus(RectsList[phase])
                frame_fea = self.getFeature(
                    corpus, description_type=self.feature_type)
                frame_label = np.asarray(FrameLabelList[phase]).reshape(-1, 1)
                print frame_fea.shape, frame_label.shape
                data = np.concatenate((frame_fea, frame_label), axis=1)
                np.save(os.path.join(self.save_floader, self.feature_type + '_' + phase + '.npy'), data)
        else:
            print 'Features existing...'
        

    def Annotation_matrixs_Generation(self, dataset_floader, trainval_test_splits, GT=False):
        """
        Args:
                GT(bool): if True, the labels of trainval and test are all groud truth, 
                            else the labels of test phase are predicted.
        """
        print 'Parsing annotation .........'
        exts = ["txt"]
        imgs = {}
        RectsList = {'trainval': [], 'test': []}
        FrameLabelList = {'trainval': [], 'test': []}
        N = {'trainval': 3493, 'test': 1337}
        frame_count = 0

        for phase in ['trainval', 'test']:
            video_id = trainval_test_splits[phase]

            if GT == False and phase == 'test':
                net = Models.resnet50_LSTM(
                    pretrained=True, model_confs=self.model_confs)
                net = net.cuda()
                net.load_state_dict(torch.load(
                    'weights/VD/action/resnet50_LSTM.pkl'))

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
                                    line)
                                FrameLabelList[phase].append(frame_label)
                                if GT == False and phase == 'test':
                                    frame_count = frame_count + 1
                                    print(
                                        "Processing Frame {}/{}".format(frame_count, 1337))
                                    img_list = sorted(glob.glob(os.path.join(
                                        dataset_floader, current_video_id, frame_id, "*.jpg")))[16:26]
                                    imgs['pre'] = img_list[:5][::-1]  # inverse
                                    imgs['back'] = img_list[4:]
                                    person_imgs_list = self.track(Rects, imgs)
                                    Rects = self.predict(frame_id, person_imgs_list, net, Rects)
                                    #print frame_id, ' ', frame_label, Rects

                                RectsList[phase].append(Rects)
                                
        return RectsList, FrameLabelList

    def AnnotationString_Parse(self, line):
        keywords = deque(line.strip().split(' '))
        frame_id = keywords.popleft().split('.')[0]
        activity = self.activity_list.index(keywords.popleft())
        Rects = []
        while keywords:
            x = int(keywords.popleft())
            y = int(keywords.popleft())
            w = int(keywords.popleft())
            h = int(keywords.popleft())
            action = int(self.action_list.index(keywords.popleft()))
            Rects.append([x, y, w, h, action])
        Rects = np.asarray(Rects)
        # sort Rects by the first col
        Rects = Rects[np.lexsort(Rects[:, ::-1].T)]
        return frame_id, activity, Rects

    def getCorpus(self, RectsList):
        # inputs: A list of Rects, such as N * array([[x, y, w, h, action],[*,*,*,*,*],...])
        # return: corpus
        corpus = []
        for Rects in RectsList:
            corpus.append(Rects[:, -1])
        return corpus

    def split_corpus(self, corpus, group_num):
        l_corpus = []
        r_corpus = []
        for doc in corpus:
            flag = len(doc) / 2
            l_corpus.append(doc[:flag])
            r_corpus.append(doc[flag:])
        return [l_corpus, r_corpus]

    def getFeature(self, corpus, description_type, group_num=2):
        Group_feature = np.zeros(
            [group_num, len(corpus), self.num_classes['action']])
        corpus_list = self.split_corpus(corpus, group_num)

        for g, corpus in enumerate(corpus_list):
            Group_feature[g] = self.get_word_vectors(corpus, description_type)
        feature = np.concatenate((Group_feature[0], Group_feature[1]), axis=1)
        return feature

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
                        feature[d][word] = self.TF_IDF(
                            word, count_list[d], count_list)
                    elif description_type == 'Bow':
                        feature[d][word] = count_list[d][word]
                    elif description_type == 'One_hot':
                        feature[d][word] = 1
        return feature

    def n_containing(self, word, count_list):
        return sum(1 for count in count_list if word in count)

    def TF_IDF(self, word, count, count_list):
        tf = float(count[word]) / sum(count.values())
        idf = math.log(len(count_list)) / \
            (1 + self.n_containing(word, count_list))
        return tf * idf

    def track(self, rects, imgs):
        person_imgs_list = []
        for i, person in enumerate(rects):
            person_imgs = []
            for phase in ['pre', 'back']:
                if phase == 'back':  # delete the middle img and inverse them
                    # print len(person_imgs)
                    person_imgs = person_imgs[1:][::-1]
                    # print len(person_imgs)

                for k, f in enumerate(imgs[phase]):
                    #print("Processing Frame {}".format(k))
                    img = io.imread(f)
                    if k == 0:
                        x, y, w, h, label = person
                        self.tracker.start_track(img, dlib.rectangle(
                            int(x), int(y), int(x + w), int(y + h)))
                    else:
                        self.tracker.update(img)
                    # save imgs
                    #cropped_image = img[Y:Y + HEIGHT,X + biasRate*WIDTH:X + (1-biasRate)*WIDTH,:]
                    pos = self.tracker.get_position()
                    top, bottom, left, right = max(int(pos.top()), 0), max(
                        int(pos.bottom()), 0), max(int(pos.left()), 0), max(int(pos.right()), 0)
                    cropped_image = img[top:bottom, left:right]
                    cropped_image = transform.resize(np.ascontiguousarray(
                        cropped_image), (224, 224), mode='constant').transpose((2, 0, 1))
                    person_imgs.append(cropped_image)

            person_imgs_list.append(person_imgs)
        return person_imgs_list

    def getBackIMG(self, T=10):
        person_imgs = []
        for t in range(T):
            none_img = '/home/ubuntu/GAR/dataset/VD/person_imgs/none.jpg'
            img = transform.resize(
                io.imread(none_img), (224, 224), mode='constant').transpose((2, 0, 1))
            person_imgs.append(img)
        #print len(person_imgs)
        return person_imgs

    def getPreds(file):
        '''
        Return:
            preds_Dicts: {frame_id:[],...} 
        '''
        lines = open(file).readlines()
        for line in lines:
            frame_id = line.split('\t')[0]
            preds = line.split('\t')[-1].split(',')
            preds_Dicts[frame_id] = preds
        return preds_Dicts

    def predict(self, frame_id, person_imgs_list, net, Rects):
        """
        Args:
            person_imgs_list: dim = num_players * 10 * 3 * 224 * 224, where K_players * 10 may less than 12
        Return:
            New_Rects:
        """
        # check for existence............
        
        if self.preds_Dicts:
            Rects[:, -1] = self.preds_Dicts[frame_id].reshape(-1)
        else:
            # complement person imgs, get a batch of inputs, note: batch =
            # K_players * 10 = 120
            truth_num_person = len(person_imgs_list)
            print 'Have %d persons' % truth_num_person
            if len(person_imgs_list) < self.num_players:
                for i in range(self.num_players - len(person_imgs_list)):
                    person_imgs_list.append(self.getBackIMG(T=10))

            imgs = np.asarray(person_imgs_list)
            imgs = imgs.reshape(-1, 3, 224, 224)
            # put imgs into CNN_model
            inputs = torch.from_numpy(imgs).float()
            # print inputs.size()
            inputs = Variable(inputs.cuda(), requires_grad=False)
            _, outputs = net.forward(inputs)
            # print outputs.data.size()
            # print torch.mean(outputs.data, 0).size()
            preds = []
            for j in range(self.num_players):
                _, pred = torch.max(torch.mean(
                    outputs.data[10 * j:10 * (j + 1)], 0).view(1, -1), 1)
                # print pred.cpu().numpy()
                preds.append(pred.cpu().numpy().tolist())

            preds = preds[:truth_num_person]
            preds = [value for pred in preds for value in pred]
            str_preds = ','.join(str(e) for e in preds)
            utils.write_txt(self.Pred_file_path, frame_id + '\t' + str_preds +'\n', 'a')
            preds = np.asarray(preds)
            
            Rects[:, -1] = preds.reshape(-1)

        return Rects
