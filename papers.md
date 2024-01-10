Awesome Papers
--------------

Papers and implementations of papers that could have use in robotics. Implementations here may not be actively developed. While implementations may often be the author's original implementation, that isn't always the case.
- [BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects](https://bundlesdf.github.io/) - 2023 - 6D pose tracking and 3D reconstruction of unknown objects
- [BundleTrack: 6D Pose Tracking for Novel Objects without Instance or Category-Level 3D Models](https://github.com/wenbowen123/BundleTrack) - 2021 - 6D object pose tracking without needing any CAD models
- [se(3)-TrackNet: Data-driven 6D Pose Tracking by Calibrating Image Residuals in Synthetic Domains](https://github.com/wenbowen123/iros20-6d-pose-tracking) - 2020 - 6D object pose tracking trained solely on synthetic data
- ["Good Robot!": Efficient Reinforcement Learning for Multi-Step Visual Tasks with Sim to Real Transfer](https://github.com/jhu-lcsr/good_robot) - 2020 - Real robot learns to complete multi-step tasks like table clearing, making stacks, and making rows in <20k simulated actions. [paper](https://arxiv.org/abs/1909.11730) (disclaimer: @ahundt is first author) [!["Good Robot!": Efficient Reinforcement Learning for Multi Step Visual Tasks via Reward Shaping](https://img.youtube.com/vi/MbCuEZadkIw/0.jpg)](https://youtu.be/MbCuEZadkIw)

- [Transporter Networks: Rearranging the Visual World for Robotic Manipulation](https://transporternets.github.io/) - [Ravens Simulator code](https://github.com/google-research/google-research/tree/master/ravens) - 2020 - Ravens is a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation, with emphasis on pick and place. It features a Gym-like API with 10 tabletop rearrangement tasks, each with (i) a scripted oracle that provides expert demonstrations (for imitation learning), and (ii) reward functions that provide partial credit (for reinforcement learning).
- [Concept2Robot: Learning Manipulation Concepts from Instructions and Human Demonstrations](https://sites.google.com/view/concept2robot) - 2020 - Language + BERT to robot actions, code TBD, pybullet sim
- [CURL: Contrastive Unsupervised Representations for RL](https://arxiv.org/abs/2004.04136) - 2020 - We use the simplest form of contrastive learning (instance-based) as an auxiliary task in model-free RL. SoTA by significant margin on DMControl and Atari for data-efficiency. 
- [Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps](https://github.com/CZ-Wu/GPNet) - 2020 - useful pybullet code robotiq gripper
- [Self-Supervised Correspondence in Visuomotor Policy Learning](https://arxiv.org/abs/1909.06933) - 2019 - [video](https://youtu.be/nDRBKb4AGmA)
- [Grasp2Vec: Learning Object Representations from Self-Supervised Grasping](https://sites.google.com/site/grasp2vec/) - 2018 - ![poster](https://pbs.twimg.com/media/Dqk8oPfWsAA96eM.jpg) no implementation available
- [Dense Object Nets: Learning Dense Visual Descriptors by and for Robotic Manipulation](https://github.com/RobotLocomotion/pytorch-dense-correspondence) - 2018 ![object feature gif](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/shoes_trim.gif)
- [Bounding Box Detection Accuracy Tradeoffs](https://arxiv.org/pdf/1611.10012.pdf) - Speed/accuracy trade-offs for modern convolutional object detectors
- [pointnet++](https://github.com/charlesq34/pointnet2) - [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](http://stanford.edu/~rqi/pointnet2/)
- [DA-RNN](https://github.com/yuxng/DA-RNN) - Semantic Mapping with Data Associated Recurrent Neural Networks
- [rpg_open_remode](https://github.com/uzh-rpg/rpg_open_remode) - This repository contains an implementation of REMODE ([REgularized MOnocular Depth Estimation](http://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf)).
- [shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org) - Fully Convolutional Networks for Semantic Segmentation, [PAMI FCN](https://arxiv.org/abs/1605.06211) and [CVPR FCN](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)
- [train-CRF-RNN](https://github.com/martinkersner/train-CRF-RNN) - Training scripts for [CRF-RNN for Semantic Image Segmentation](https://github.com/torrvision/crfasrnn).
- [train-DeepLab](https://github.com/martinkersner/train-DeepLab) - Scripts for training [DeepLab for Semantic Image Segmentation](https://bitbucket.org/deeplab/deeplab-public) using [strongly](https://github.com/martinkersner/train-DeepLab#strong-annotations) and [weakly annotated data](https://github.com/martinkersner/train-DeepLab#weak-annotations). [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](http://arxiv.org/abs/1412.7062) and [Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation](http://arxiv.org/abs/1502.02734) papers describe training procedure using strongly and weakly annotated data, respectively.
- [text_objseg](https://github.com/ronghanghu/text_objseg) Segmentation from Natural Language Expressions
- [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), Asynchronous Advantage Actor Critic (A3C)
    - [tensorpack/examples/A3C-Gym](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/A3C-Gym) - Multi-GPU version of the A3C algorithm.
    - [ga3c](https://github.com/NVlabs/GA3C) - Hybrid CPU/GPU implementation of the A3C algorithm for deep reinforcement learning.

### Robotic Grasping

- [Real-Time Grasp Detection Using Convolutional Neural Networks](https://arxiv.org/pdf/1412.3128.pdf) - (2015)
- [Dex-Net 2.0](https://arxiv.org/pdf/1703.09312.pdf) - Dex-Net 2.0: Deep Learning to Plan Robust
Grasps with Synthetic Point Clouds and Analytic Grasp Metrics
- [Multi-task Domain Adaptation for Deep Learning of Instance Grasping from Simulation](https://arxiv.org/pdf/1710.06422.pdf) - (2017)
- [End-to-End Learning of Semantic Grasping](https://arxiv.org/pdf/1707.01932.pdf) - Paper from google, uses hand eye coordination methods + classification of what object would be grasped
- [Robotic Grasp Detection using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1611.08036.pdf) - (2017) uses 2 resnets
- [Convolutional Residual Network for Grasp Localization](http://www2.ift.ulaval.ca/~pgiguere/papers/ResNetGraspCRV2017.pdf) - (2017) uses 1 resnet
- [Supervision via Competition: Robot Adversaries for Learning Tasks](https://arxiv.org/pdf/1610.01685v1.pdf) - (2017 with [Code](https://github.com/lerrel/Grasp-Detector)) One robot holds an object and tries to make the object difficult to grasp.

#### Older papers useful as references for the above

- [Deep Learning for Detecting Robotic Grasps](http://pr.cs.cornell.edu/papers/lenz_ijrr2014_deepgrasping.pdf) - (2014) Created the [Cornell Graping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)
- [Efficient Grasping from RGBD Images: Learning using a new
Rectangle Representation](http://pr.cs.cornell.edu/grasping/jiang_rectanglerepresentation_fastgrasping.pdf) - (2011) Defines the rectangle representation utilized in many of the above grasping papers.


## Reinforcement learning (including non-robotics)

- [Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/abs/2006.00979) - 2020 - Deepmind
- [RECURRENT EXPERIENCE REPLAY IN DISTRIBUTED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=r1lyTjAqYX) - 2019 - Recurrent Replay Distributed DQN (R2D2) - Deepmind
- [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.13888.pdf) - 2020 - Offline RL benchmark - Deepmind
- [Making Efficient Use of Demonstrations to Solve Hard Exploration Problems](https://arxiv.org/pdf/1909.01387.pdf) - 2019 - R2D3 - Deepmind

## Language and Robots

- [Concept2Robot: Learning Manipulation Concepts from Instructions and Human Demonstrations](http://www.roboticsproceedings.org/rss16/p082.pdf) - RSS2020
