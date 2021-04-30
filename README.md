# pytorch-DEM
double-embedding

## Introduction
Multi-person pose estimation is a fundamental and challenging problem to many computer vision tasks. Most existing methods can
be broadly categorized into two classes: top-down and bottom-up methods. Both of the two types of methods involve two stages, namely, person
detection and joints detection. Conventionally, the two stages are implemented separately without considering their interactions between them,
and this may inevitably cause some issue intrinsically. In this paper, we
present a novel method to simplify the pipeline by implementing person detection and joints detection simultaneously. We propose a Double
Embedding (DE) method to complete the multi-person pose estimation
task in a global-to-local way. DE consists of Global Embedding (GE)
and Local Embedding (LE). GE encodes different person instances and
processes information covering the whole image and LE encodes the local limbs information. GE functions for the person detection in top-down
strategy while LE connects the rest joints sequentially which functions
for joint grouping and information processing in A bottom-up strategy.
Based on LE, we design the Mutual Refine Machine (MRM) to reduce
the prediction difficulty in complex scenarios. MRM can effectively realize the information communicating between keypoints and further improve the accuracy. We achieve the competitive results on benchmarks
MSCOCO, MPII and CrowdPose, demonstrating the effectiveness and
generalization ability of our method.

### Acknowledge
Our code is mainly based on [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). 

## Quick start
### Installation
1. Install pytorch >= v1.1.0 following [official instruction](https://pytorch.org/).  
   - **Tested with pytorch v1.4.0**
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
   - **There is a bug in the CrowdPoseAPI, please reverse https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
5. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

#### Mixed-precision training
Due to large input size for bottom-up methods, we use mixed-precision training to train our network by using the following command:
```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    FP16.ENABLED True FP16.DYNAMIC_LOSS_SCALE True
```

#### Synchronized BatchNorm training
If you have limited GPU memory, please try to reduce batch size and use SyncBN to train the network by using the following command:
```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    FP16.ENABLED True FP16.DYNAMIC_LOSS_SCALE True \
    MODEL.SYNC_BN True
```

Our code for mixed-precision training is also borrowed from [NVIDIA Apex API](https://github.com/NVIDIA/apex).

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{Xu2020bottom,
  title={A Global to Local Double Embedding Method for Multi-person Pose Estimation},
  author={Yiming Xu and Jiaxin Li and Yiheng Peng and Yan Ding and HuaLiang Wei},
  year={2020}
}

@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}
````
