# Participation-Contributed Temporal Dynamic Model for Group Activity Recognition. [PDF](https://www.researchgate.net/profile/Rui_Yan31/publication/328372578_Participation-Contributed_Temporal_Dynamic_Model_for_Group_Activity_Recognition/links/5bed27684585150b2bb79e69/Participation-Contributed-Temporal-Dynamic-Model-for-Group-Activity-Recognition.pdf)

We give a general **DMS**(**D**ata, **M**odel, **S**olver) code framework for PCTDM, impelemented by Pytorch. You can apply new model or dataset into this framework by modifying the files in `Configs` easily! For further information about me, welcome to my [homepage](https://ruiyan1995.github.io/).


## Requirements
&gt; Ubuntu 16.04  
&gt; pytorch 0.4.1  
= python 2.7  
`pip install dlib`

## The general piplines of GAR
You can run `python GAR.py` to excute all the following steps.
### Step Zero: Preprocessing dataset
- To download [VD](https://github.com/mostafa-saad/deep-activity-rec#dataset) and [CAD](http://vhosts.eecs.umich.edu/vision//activity-dataset.html) at './dataset/VD' and './dataset/CAD' folder;
- Add `none.jpg`
- To track the persons and generate the train/test files by using **Preprocessing.py**;

### Step One: Action Level
- To create a `Piplines` instance as:

&nbsp;&nbsp;`Action = Action_Level(dataset_root, dataset_name, 'trainval_action')`;
- For action recognition, you can use `Action.trainval()`;
- For extracting action features, you can use `Action.extract_feas(save_folder='*')`.

### Step Two: Activity Level
This is the core part of GAR which need your design. We proposed a novel PCTDM to aggreate the action features with attending to key persons.
- To create a `Piplines` instance as:

&nbsp;&nbsp;`Activity = Activity_Level(dataset_root, dataset_name, 'trainval_activity')`;
- For activity recognition, you can use `Activity.trainval()`.

### Step Three: Evaluate
- To show some results at activity level, you can use `Activity.evaluate()`.

All steps may take about 15 hours for 'VD', and 5 hours for 'CAD'.

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

## Contact Information
Feel free to create a pull request or contact me by Email = ["ruiyan", at, "njust", dot, "edu", dot, "cn"], if you find any bugs. 