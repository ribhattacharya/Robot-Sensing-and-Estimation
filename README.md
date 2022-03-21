# Robot-Sensing-and-Estimation
Projects involving sensing/estimation of the environment and location of robots
This repository contains 3 folders/projects

## Color Classification and Recycling Bin Detection
- Trained a probabilistic color model from given pixel data to distinguish among red, green, and blue pixels
- Trained a probabilistic color model to recognize recycling-bin specific blue color and used it to segment unseen images into blue regions. Given the blue regions, detected blue recycling bins and formed a bounding box around each one

## Particle Filter SLAM
- Implemented simultaneous localization and mapping (SLAM) using odometry, 2-D LiDAR scans, and stereo camera measurements from an autonomous car. Used the odometry and LiDAR measurements to localize the robot and build a 2-D occupancy grid map of the environment. 
- The goal of this project is to use a particle filter with a differential-drive motion model and scan-grid correlation observation model for simultaneous localization and occupancy-grid mapping.

## Visual Inertial SLAM (EKF)
- Implemented visual-inertial simultaneous localization and mapping (SLAM) using an extended Kalman filter (EKF)
- Used synchronized measurements from an inertial measurement unit (IMU) and a stereo camera
- Implemented EKF prediction + update steps with odometry measurements and EKF update using enviropnment data (stereo images) 
