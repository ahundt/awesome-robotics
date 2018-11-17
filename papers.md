Awesome Papers
--------------

Papers and implementations of papers that could have use in robotics. Implementations here may not be actively developed. While implementations may often be the author's original implementation, that isn't always the case.

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
