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
the prediction difficulty in complex scenarios. MRM can effectively realize the information communicating between keypoints and further improve the accuracy. We achieve the competitive results on benchmarks
MSCOCO, MPII and CrowdPose, demonstrating the effectiveness and
generalization ability of our method.

### Acknowledge
Our code is mainly based on [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). 

