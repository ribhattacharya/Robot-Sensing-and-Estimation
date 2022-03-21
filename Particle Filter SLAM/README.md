# Color Classification and Recycling Bin Detection
(Cmd+Shift+V to go into markdown reader mode in Mac)
(Ctrl+Shift+V to go into markdown reader mode in Windows)

The .py script is organized as follows;

## class PartFilterSLAM 
1. **__init__**: initializes all parameters needed for the particle filter; descriptions of parameters given in the code comments; parameters can be changed here
2. **read_data_from_csv**: function to read data from .csv files; seperates data and timestamps
3. **comp_inputs**: computes encoder velocity and angular velocity from encoder & fog data; stores it in a file
4. **robot_path**: calculate update steps from encoder and FOG data, to be used for dead reckoning
5. **bresenham2D**: bresenham2D algorithm
6. **mapCorrelation**: find map correlation b/w 2 grids
7. **lidar_2_vehicle_frame**: convert lidar (polar coordinate data) to vehicle cartesian data
8. **sigmoid**: returns the sigmoid values of a given input
9. **map_log_odds**: compute and return the modified log-odds map
10. **update_bitmap**: update occupancy grid map based on map log odds value
11. **__name__ == '__main__'**: Mapping, prediction, update loop for computing the occupancy grid map

## Files reqd

1. **input_data.npy**: stores the linear velocity & angular velocity data, to be kept in the same folder (*root_folder\input_data.npy*) as the script (*root_folder\part_slam.py*)
2. **lidar.csv**: lidar data, to be stored as *root_folder\sensor_data\lidar.csv*
3. **(Optional) encoder/fog.csv**: Both data files have been parsed and the v/w have been calculated in *input_data.npy*. But in case the *compute_inputs()* function has to be run, these files are required in the same folder as lidar.csv (*root_folder\sensor_data\encoder.csv* and *root_folder\sensor_data\fog.csv*)

## Results
<a href="url"><img src="https://github.com/ribhattacharya/Robot-Sensing-and-Estimation/blob/main/Particle%20Filter%20SLAM/results/Figure_1.png" align="left" height="500" ></a>
<a href="url"><img src="https://github.com/ribhattacharya/Robot-Sensing-and-Estimation/blob/main/Particle%20Filter%20SLAM/results/Figure_2.png" align="right" height="500" ></a>
<a href="url"><img src="https://github.com/ribhattacharya/Robot-Sensing-and-Estimation/blob/main/Particle%20Filter%20SLAM/results/Figure_3.png" align="left" height="500" ></a>
<a href="url"><img src="https://github.com/ribhattacharya/Robot-Sensing-and-Estimation/blob/main/Particle%20Filter%20SLAM/results/Figure_4.png" align="right" height="500" ></a>
