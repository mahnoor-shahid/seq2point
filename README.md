
# Implementation of Seq2Point model for Energy Disaggregation
> Ready and easy to implement a complete end-to-end machine learning pipeline for training the archiecture of seq2point model for energy disaggregation. This project uses refit_loader as a submodule which has taken the advantage of **Dask Dataframes** to ease and fasten the process of loading all the data of REFIT dataset and also provides some data transformation functionalities. 



### Sequence-to-point learning with neural networks for nonintrusive load monitoring

In this research, authors have proposed sequence-to-point learning, where the input is a window of the mains and the output is a single point of the target appliance. They have used convolutional neural networks to train the model. They showed that the convolutional neural networks can inherently learn the signatures of the target appliances, which are automatically added into the model to reduce the identifiability problem and haved showed that the method achieve state-of-the-art performance. <br />
~ Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, Charles Sutton  <br />

### Research Paper
For more detail information, visit the following link: <br />
https://arxiv.org/abs/1612.09106 <br />

## Dependencies
Ensure that the following dependencies are satisfied either in your current environment 
```
  - python=3.9.2
  - numpy=1.23.3
  - pandas=1.4.2
  - dask=2021.06.2
  - json=2.0.9
  - torch=1.12.1
  - tensorboard=2.10.0
```
or create a new environment using 'environment.yml'
```
conda create env --file=environment.yml
conda activate seq2point_env
```

# Steps to implement this project
1) Clone this project into your specified target directory.
```
git clone https://github.com/mahnoor-shahid/seq2point
```

2) This project uses nilm-analyzer python package. Use the following command to install the package.
```
pip install nilm-analyzer
```

3) [Download](#downloads) the refit dataset and it must be located in the data folder as specified by config.json file of refit_loader. (If not sure where data folder should be located, take a reference from [repository structure](#repository_structure))
```
{ 
    "DATA_FOLDER" : "data/refit/",
    "DATA_TYPE" : ".csv",
    "README_FILE" : "refit_loader/REFIT_Readme.txt",
    "REFIT_HOUSES" : [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21]
}
```

4) All configurations are located in 'configs' folder of this project. You can change the configurations according to your experiment. <br /> Majorly 'training_config.json' sets how you want to run your experiment.
```
{
    "TARGET_APPLIANCE" : "KETTLE",
    "TARGET_HOUSES":
        {
            "TRAIN" : [2],
            "VALIDATE" : [2],
            "TEST" : [2]
        },
    "SPLIT_PROPORTION" :
        {
            "TRAIN_PERCENT" : 0.6,
            "VALIDATE_PERCENT" : 0.2,
            "TEST_PERCENT" : 0.2
        },
    "THRESHOLD": 0.1,
    "SUBSET_DAYS": 1,
    "NORMALIZE": "Standard",

    "TRAIN_BATCH_SIZE" : 64,
    "VALIDATION_BATCH_SIZE" : 64,
    "TEST_BATCH_SIZE" : 64,
    "LEARNING_RATE" : 0.001,
    "NUM_EPOCHS" : 12,
    "OPTIMIZER": "optim.Adam",
    "LOSS": "nn.MSELoss",
    "LOSS_REDUCTION": "mean",
    "EARLY_STOPPING_THRESHOLD": 6,
    "PRE_TRAINED_MODEL_FLAG": false
}
```
5) Simply execute the main.ipynb file to run your experiment. The following line creates a folder for every experiment that you want to run and it stores all the outputs, models, plots generated during that experiment in that experiments folder.
```
TRAINING_CONFIG['EXPERIMENT_PATH'] = f'experiments/{TRAINING_CONFIG["TARGET_APPLIANCE"]}/{TRAINING_CONFIG["TARGET_HOUSES"]["TRAIN"]}/'
```


## Repository_Structure
This repository follows the below structure format:
```
|
|  
├── refit_loader
|  └── data_loader.py
|  └── utilities
|   |  └── configuration.py
|   |  └── parser.py
|   |  └── time_utils.py
|   |  └── validations.py
|  └── config.json
|  └── environment.yml
|  └── REFIT_README.txt
|  └── readme.md
|
|
├── configs
|  └── dataset_config.json
|  └── model_config.json
|  └── plot_config.json
|  └── training_config.json
|
|
├── data
|  └── refit
|  |  └── REFIT_Readme.txt
|  |  └── House_1.csv
|  |  └── House_2.csv
|  |  └── House_3.csv
|  |  └── House_4.csv
|  |  └── House_5.csv
|  |  └── House_6.csv
|  |  └── House_7.csv
|  |  └── House_8.csv
|  |  └── House_9.csv
|  |  └── House_10.csv
|  |  └── House_11.csv
|  |  └── House_12.csv
|  |  └── House_13.csv
|  |  └── House_15.csv
|  |  └── House_16.csv
|  |  └── House_17.csv
|  |  └── House_18.csv
|  |  └── House_19.csv
|  |  └── House_20.csv
|
|
├── dataset_management
|  └── dataloader.py
|  └── generator.py
|
|
├── seq2point
|  └── seq2point.py
|
|
├── training
|  └── train.py
|
|
├── utils
|  └── compute_metrics.py
|  └── plotting_traces.py
|  └── configuration.py
|  └── training_utilities.py
|
|
├── expermiments
|
|
├── main.ipynb
├── expermiments_report.ipynb
|
|
├── environment.yml
├── .gitmodules
├── .gitignore
├── readme.md
|
|
```

## Downloads
The REFIT Smart Home dataset is a publicly available dataset of Smart Home data. <br />
Dataset - https://pureportal.strath.ac.uk/files/52873459/Processed_Data_CSV.7z <br />


## Citations
```
@article{https://doi.org/10.48550/arxiv.1612.09106,
  doi = {10.48550/ARXIV.1612.09106},
  
  url = {https://arxiv.org/abs/1612.09106},
  
  author = {Zhang, Chaoyun and Zhong, Mingjun and Wang, Zongzuo and Goddard, Nigel and Sutton, Charles},
  
  keywords = {Applications (stat.AP), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Sequence-to-point learning with neural networks for nonintrusive load monitoring},
  
  publisher = {arXiv},
  
  year = {2016},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

