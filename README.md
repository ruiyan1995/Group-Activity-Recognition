# Participation-Contributed Temporal Dynamic Model for Group Activity Recognition.

This repository includes the code of PCTDM and some baselines such as HDTM[1], Wang[2], impelemented by Pytorch. We give a general DMS code framework for Group Activity Recognition task. You can apply new model or dataset into this framework easily! In 2019, I will clear up the code again! For further information about me, you can go to my [homepage](https://ruiyan1995.github.io/).


## The general piplines of GAR
You can run `python GAR.py` to excute all the following steps.
### Step Zero: Preprocessing dataset
- To download VD and CAD;
- To track the persons in video by Dlib, which implemented in **Preprocessing.py**;

### Step One: Action Level
- To create a `Piplines` instance as:
`Action = Action_Level(dataset_root, dataset_name, 'trainval_action')`;
- For action recognition, you can use `Action.trainval()`;
- For extracting action features, you can use `Action.extract_feas(save_folder)`;

### Step Two: Activity Level
This is the core part of GAR which need your design. We proposed a novel PCTDM to aggreate the action features with attending to key persons.
- For action features aggregation, you can use 
- For activity recognition

## License and Citation 
Please cite the following paper in your publications if it helps your research.

@inproceedings{yan2018participation,  
&nbsp;&nbsp;&nbsp;&nbsp;title={Participation-Contributed Temporal Dynamic Model for Group Activity Recognition},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Yan, Rui and Tang, Jinhui and Shu, Xiangbo and Li, Zechao and Tian, Qi},  
&nbsp;&nbsp;&nbsp;&nbsp;booktitle={2018 ACM Multimedia Conference on Multimedia Conference},  
&nbsp;&nbsp;&nbsp;&nbsp;pages={1292--1300},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2018},  
&nbsp;&nbsp;&nbsp;&nbsp;organization={ACM}  
}
> [1] ddd

> [2] ddd