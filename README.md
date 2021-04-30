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

