# Awesome Robotics

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Awesome links, software libraries, [papers](papers.md), and other [intersting links](interesting.md) that are useful for robots.


Relevant Awesome Lists
----------------------

- [Kiloreaux/awesome-robotics](https://github.com/Kiloreux/awesome-robotics) - Learn about Robotics.
- [Robotics Libraries](https://github.com/jslee02/awesome-robotics-libraries) - Another list of awesome robotics libraries.
- [Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
- [Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning) - Neural networks.
    - [TensorFlow](https://github.com/jtoy/awesome-tensorflow) - Library for machine intelligence.
    - [Papers](https://github.com/terryum/awesome-deep-learning-papers) - The most cited deep learning papers.
- [Deep Vision](https://github.com/kjw0612/awesome-deep-vision) - Deep learning for computer vision
- [Data Visualization](https://github.com/fasouto/awesome-dataviz) - See what your robot is doing with any programming language.

Simulators
----------

- [V-REP](coppeliarobotics.com/index.html) - Create, Simulate, any Robot.
- [Microsoft Airsim](https://github.com/Microsoft/AirSim) - Open source simulator based on Unreal Engine for autonomous vehicles from Microsoft AI & Research.
- [Bullet Physics SDK](https://github.com/bulletphysics/bullet3) - Real-time collision detection and multi-physics simulation for VR, games, visual effects, robotics, machine learning etc. Also see [pybullet](https://pybullet.org).

Visualization, Video, Display, and Rendering
-----------------------

 - [Pangolin](https://github.com/stevenlovegrove/Pangolin) - A lightweight portable rapid development library for managing OpenGL display / interaction and abstracting video input.
- [PlotJuggler](https://github.com/facontidavide/PlotJuggler) - Quickly plot and re-plot data on the fly! Includes optional ROS integration.
- [Data Visualization](https://github.com/fasouto/awesome-dataviz) - A list of awesome data visualization tools.

Machine Learning
----------------

### TensorFlow related

- [Keras](keras.io) - Deep Learning library for Python. Convnets, recurrent neural networks, and more. Runs on TensorFlow or Theano.
- [keras-contrib](https://github.com/farizrahman4u/keras-contrib) - Keras community contributions.
- [TensorFlow](tensorflow.org) - An open-source software library for Machine Intelligence.
- [recurrentshop](https://github.com/datalogai/recurrentshop) - Framework for building complex recurrent neural networks with Keras.
- [tensorpack](https://github.com/ppwwyyxx/tensorpack) - Neural Network Toolbox on TensorFlow.
- [tensorlayer](https://github.com/zsdonghao/tensorlayer) - Deep Learning and Reinforcement Learning Library for Researchers and Engineers.
- [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) - TensorFlow Tutorial and Examples for beginners.
- [hyperas](https://github.com/maxpumperla/hyperas) - Keras + Hyperopt: A very simple wrapper for convenient hyperparameter optimization.
- [elephas](https://github.com/maxpumperla/elephas) - Distributed Deep learning with Keras & Spark
- [PipelineAI](https://github.com/fluxcapacitor/pipeline) - End-to-End ML and AI Platform for Real-time Spark and Tensorflow Data Pipelines.
- [sonnet](https://github.com/deepmind/sonnet) - Google Deepmind APIs on top of TensorFlow.
- [visipedia/tfrecords](https://github.com/visipedia/tfrecords) - Demonstrates the use of TensorFlow's TFRecord data format.

#### Image Segmentation

- [tf-image-segmentation](https://github.com/warmspringwinds/tf-image-segmentation) - Image Segmentation framework based on Tensorflow and TF-Slim library.
- [Keras-FCN](https://github.com/aurora95/Keras-FCN)


Logging and Messaging
---------------------

- [spdlog](https://github.com/gabime/spdlog) - Super fast C++ logging library.
- [lcm](https://github.com/lcm-proj/lcm) - Lightweight Communications and Marshalling, message passing and data marshalling for real-time systems where high-bandwidth and low latency are critical.

Tracking
--------

- [simtrack](https://github.com/karlpauwels/simtrack) - A simulation-based framework for tracking.
- [ar_track_alvar](https://github.com/sniekum/ar_track_alvar) - AR tag tracking library for ROS.
- [artoolkit5](https://github.com/artoolkit/artoolkit5) - Augmented Reality Toolkit, which has excellent AR tag tracking software.

Robot Operating System (ROS)
----------------------------

- [ROS](ros.org) - Main ROS website.
- [ros2/design](https://github.com/ros2/design) - Design documentation for ROS 2.0 effort.


Kinematics, Dynamics, Constrained Optimization
----------------------------------------------

- [jrl-umi3218/Tasks](https://github.com/jrl-umi3218/Tasks) - Tasks is library for real time control of robots and kinematic trees using constrained optimization.
- [jrl-umi3218/RBDyn](https://github.com/jrl-umi3218/RBDyn) - RBDyn provides a set of classes and functions to model the dynamics of rigid body systems.
- [ceres-solver](https://github.com/ceres-solver/ceres-solver) - Solve Non-linear Least Squares problems with bounds constraints and general unconstrained optimization problems. Used in production at Google since 2010.
- [orocos_kinematics_dynamics](https://github.com/orocos/orocos_kinematics_dynamics) - Orocos Kinematics and Dynamics C++ library.
- [flexible-collsion-library](https://github.com/flexible-collision-library/fcl) - Performs three types of proximity queries on a pair of geometric models composed of triangles, integrated with ROS. 
- [robot_calibration](https://github.com/mikeferguson/robot_calibration) - generic robot kinematics calibration for ROS

Calibration
-----------

- [handeye-calib-camodocal](https://github.com/jhu-lcsr/handeye_calib_camodocal) - generic robot hand-eye calibration.
- [robot_calibration](https://github.com/mikeferguson/robot_calibration) - generic robot kinematics calibration for ROS
- [kalibr](https://github.com/ethz-asl/kalibr) - camera and imu calibration for ROS

Reinforcement Learning
----------------------

- [TensorForce](https://github.com/reinforceio/tensorforce) - A TensorFlow library for applied reinforcement learning
- [gqcnn](https://github.com/BerkeleyAutomation/gqcnn) -  [Grasp Quality Convolutional Neural Networks (GQ-CNNs)](https://berkeleyautomation.github.io/gqcnn/info/info.html) for grasp planning using training datasets from the [Dexterity Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net)
- [Guided Policy Search](https://github.com/cbfinn/gps) - Guided policy search (gps) algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work. 

Drivers for Sensors, Devices and Arms
-------------------------------------

- [libfreenect2](https://github.com/OpenKinect/libfreenect2) - Open source drivers for the Kinect for Windows v2 and Xbox One devices.
- [iai_kinect2](https://github.com/code-iai/iai_kinect2) - Tools for using the Kinect One (Kinect v2) in ROS.
- [grl](https://github.com/ahundt/grl) - Generic Robotics Library: Cross platform drivers for Kuka iiwa and Atracsys FusionTrack with optional v-rep and ros drivers. Also has cross platform Hand Eye Calibration and Tool Tip Calibration.

Datasets
--------

- [CoSTAR Block Stacking Dataset](https://sites.google.com/site/costardataset) - Robot stacking colored children's blocks (disclaimer: created by @ahundt)
- [shapestacks](http://shapestacks.robots.ox.ac.uk/#paper) - simulated stacks of colored children's objects
- [pascal voc 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) - The classic reference image segmentation dataset.
- [openimages](https://github.com/openimages/dataset/) - Huge imagenet style dataset by Google.
- [COCO](mscoco.org) - Objects with segmentation, keypoints, and links to many other external datasets.
- [cocostuff](https://github.com/nightrome/cocostuff) - COCO additional full scene segmentation including backgrounds and annotator.
- [Google Brain Robot Data](https://sites.google.com/site/brainrobotdata/home) - Robotics datasets including grasping, pushing, and pouring.
- [Materials in Context](http://opensurfaces.cs.cornell.edu/publications/minc/) - Materials Dataset with real world images in 23 categories.
- [Dex-Net 2.0](http://bair.berkeley.edu/blog/2017/06/27/dexnet-2.0/) - 6.7 million pairs of synthetic point clouds and grasps with robustness labels.

#### Dataset Collection

- [cocostuff](https://github.com/nightrome/cocostuff) - COCO additional full scene segmentation including backgrounds and annotator.

Linear Algebra & Geometry
-------------------------

- [Eigen](eigen.tuxfamily.org) - Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
- [Boost.QVM](https://github.com/boostorg/qvm) - Quaternions, Vectors, Matrices library for Boost.
- [Boost.Geometry](https://github.com/boostorg/geometry/) - Boost.Geometry contains instantiable geometry classes, but library users can also use their own.
- [SpaceVecAlg](https://github.com/jrl-umi3218/SpaceVecAlg) - Implementation of spatial vector algebra for 3D geometry with the Eigen3 linear algebra library.
- [Sophus](https://github.com/strasdat/Sophus) - C++ implementation of Lie Groups which are for 3D Geometry, using Eigen.


Point Clouds
------------

- [libpointmatcher](https://github.com/ethz-asl/libpointmatcher) - An "Iterative Closest Point" library robotics and 2-D/3-D mapping.
- [Point Cloud Library (pcl)](https://github.com/PointCloudLibrary/pcl) - The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing.



Simultaneous Localization and Mapping (SLAM)
--------------------------------------------

- [ElasticFusion](https://github.com/mp3guy/ElasticFusion) - Real-time dense visual SLAM system.
- [co-fusion](https://github.com/martinruenz/co-fusion) - Real-time Segmentation, Tracking and Fusion of Multiple Objects. Extends ElasticFusion.
- [Google Cartographer](https://github.com/googlecartographer/cartographer/) - Cartographer is a system that provides real-time simultaneous localization and mapping (SLAM) in 2D and 3D across multiple platforms and sensor configurations.
- [OctoMap](https://github.com/OctoMap/octomap) - An Efficient Probabilistic 3D Mapping Framework Based on Octrees. Contains the main OctoMap library, the viewer octovis, and dynamicEDT3D.
- [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) - Real-Time SLAM for Monocular, Stereo and RGB-D Cameras, with Loop Detection and Relocalization Capabilities.


# License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
