# Group-Activity-Recognition
Participation-Contributed Temporal Dynamic Model for Group Activity Recognition.

This repository includes the code of PCTDM and some baselines such as HDTM[1], Wang[2], impelemented by Pytorch. We give a general DMS code framework for Group Activity Recognition task. You can apply new model or new dataset into this framework easily! In 2019, I will clear up the code again! For further information about me, you can go to my [homepage](https://ruiyan1995.github.io/)

You can run it as following scripts:
python GAR.py

The general piplines of GAR:

Preprocessing dataset

To download VD and CAD;

To track the persons in video by Dlib, which implemented in Preprocessing.py;

Action Level (action recognition and extracting action features)
To create a Piplines instance Action = Action_Level(dataset_root, dataset_name, 'trainval_action');
For action recognition, you can run Action.trainval();
For extracting action features, you can run Action.extract_feas(save_folder);

Activity Level (action features aggregation and activity recognition)



Pealse cite the following paper:  

@inproceedings{yan2018participation,  
&nbsp;&nbsp;&nbsp;&nbsp;title={Participation-Contributed Temporal Dynamic Model for Group Activity Recognition},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Yan, Rui and Tang, Jinhui and Shu, Xiangbo and Li, Zechao and Tian, Qi},  
&nbsp;&nbsp;&nbsp;&nbsp;booktitle={2018 ACM Multimedia Conference on Multimedia Conference},  
&nbsp;&nbsp;&nbsp;&nbsp;pages={1292--1300},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2018},  
&nbsp;&nbsp;&nbsp;&nbsp;organization={ACM}  
}
