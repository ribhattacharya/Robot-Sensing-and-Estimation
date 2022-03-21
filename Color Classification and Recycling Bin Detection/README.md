# Color Classification and Recycling Bin Detection
(Cmd+Shift+V to go into markdown reader mode in Mac)
(Ctrl+Shift+V to go into markdown reader mode in Windows)

The folder structure is divided into two folders.

## pixel_classification 
1. **generate_rgb_data.py**: generates the data and saves into a file
2. **pixel_classifier.py**: main file containing the GDA classifier
3. **pixel_training.py**: training code for learning parameters
4. **trained_mu.npy**: mean data file
5. **trained_sigma.npy**: covariance data file
6. **trained_theta.npy**: marginal pdf file
    
    
## bin_detection
1. **data_collection.py**: collects data from each image using roipoly; implemented a loop     to     continuously parse through each image without having to open each manually. It also gives you an option to revisit an image in case more pixels or different label pixels need to be collected. All the data is collated and unique data points are appended after each file is parsed.
2. **bin_color_training.py**: file used for learning the parameters using GDA
3. **bin_detector.py**: main file containing segmentation and bin detection code
4. **generate_figs.py**: file used for generating figures in the report, and occasional debuging as well
5. **trained_mu.npy**: mean data file
6. **trained_sigma.npy**: covariance data file
7. **trained_theta.npy**: marginal pdf file

For classification and detection, only the **pixel_classifier.py** and the **bin_detector.py** files are required, along with their respective training parameters **trained_mu.npy**, **trained_sigma.npy** and **trained_theta.npy**. All other files are made available for reference purposes. Please contact me in case of any issues.


Rishabh Bhattacharya  
Mechanical and Aerospace Engineering  
ribhattacharya@ucsd.edu



