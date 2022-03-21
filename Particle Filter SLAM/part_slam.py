# # Function file for PR2

# ## Import all packages
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import os
matplotlib.use('Qt5Agg')


class PartFilterSLAM():
  def __init__(self):
    # initialize MAP
    self.MAP = {}
    self.MAP['res']   = 1 #meters
    self.MAP['xmin']  = -1500  #meters
    self.MAP['ymin']  = -1500
    self.MAP['xmax']  =  1500
    self.MAP['ymax']  =  1500
    self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
    self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
    self.MAP['logmap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']), dtype = np.float64)
    self.MAP['bitmap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']), dtype = np.int16)

    # particle filter parameters
    self.lidar_confidence  = 4                           # Lidar confidence (80%)
    self.z_thresh = 1.5                                  # Z threshold for lidar to vehicle transform (m)
    self.N_particles = 30                                # No. of particles
    self.ratio = 20                                    # prediction to update ratio (for every 1 update, we have 'ratio' predictions)
    self.v_var, self.w_var = 1e-1, 1e-3                       # Variances for gaussian noise in inputs
    self.resample_thresh = np.floor(0.6 * self.N_particles)   # 60% of original particles


# ## Read data function
  def read_data_from_csv(self, filename):
    '''
    INPUT 
    filename        file address

    OUTPUT 
    timestamp       timestamp of each observation
    data            a numpy array containing a sensor measurement in each row
    '''
    data_csv = pd.read_csv(filename, header=None)
    data = data_csv.values[:, 1:]
    timestamp = data_csv.values[:, 0]
    return timestamp, data


# ## Compute Inputs
# Computes encoder velocity and angular velocity from encoder & fog data; stores it in a file

  def comp_inputs(self):

      # working on encoder data
      folder_path = os.path.dirname(os.path.abspath(__file__)) 
      enc_timestamp, enc_tick = self.read_data_from_csv(os.path.join(folder_path, 'sensor_data/encoder.csv'))
      enc_res = 4096
      enc_dia = np.array([0.623479, 0.622806]) #in meters

      # converting values to standard units ('s' & 'm') 
      enc_timestamp = enc_timestamp/(1e9)
      enc_dist = np.diff(enc_tick, axis = 0) * enc_dia * np.pi / enc_res

      del_t = np.diff(enc_timestamp)

      v = np.sum(enc_dist, axis = 1) * 0.5 / del_t

      # working on FOG data
      _, fog_data = self.read_data_from_csv(os.path.join(folder_path, 'sensor_data/fog.csv'))
      fog_yaw = fog_data[:,2] #keeping only the yaw angle (in rad)


      yaw = np.zeros_like(del_t)
      w = np.zeros_like(del_t)

      for i in range(len(del_t)):
          yaw[i] = np.sum(fog_yaw[: 10*i + 10]) #cumulative sum
          w[i] = np.sum(fog_yaw[10*i : 10*i+10]) / del_t[i]

      # [delta t, velocity, angular velocity]
      X = np.array([del_t, v, w]).T

      # add (0,0,0) state to the x matrix
      X = np.insert(X, 0, np.array([0, 0, 0]).T, axis = 0)

      np.save(os.path.join(folder_path, 'input_data.npy'), X)
    


# ## Robot path calculate (dead reckoning)
# Calculate update steps from encoder and FOG data, to be used for dead reckoning

  def robot_path(self):
      # load data

      folder_path = os.path.dirname(os.path.abspath(__file__)) 
      data = np.load(os.path.join(folder_path, 'input_data.npy'))
      del_t = data[:,0]
      v = data[:,1]
      w = data[:,2]
      yaw = np.cumsum(w * del_t)

      # differential drive model at each timestamp 
      X = np.array(del_t * [v * np.cos(yaw), v * np.sin(yaw), w]).T

      # cumulative sum to get state x_t+1 
      X = np.cumsum(X, axis = 0)

      return X


# ## LIDAR transforms

# ### Bresenham 2D algorithm 

  def bresenham2D(self, sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
      (sx, sy)	start point of ray
      (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
      dx,dy = dy,dx # swap 

    if dy == 0:
      q = np.zeros((dx+1,1))
    else:
      q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
      if sy <= ey:
        y = np.arange(sy,ey+1)
      else:
        y = np.arange(sy,ey-1,-1)
      if sx <= ex:
        x = sx + np.cumsum(q)
      else:
        x = sx - np.cumsum(q)
    else:
      if sx <= ex:
        x = np.arange(sx,ex+1)
      else:
        x = np.arange(sx,ex-1,-1)
      if sy <= ey:
        y = sy + np.cumsum(q)
      else:
        y = sy - np.cumsum(q)
    return np.vstack((x,y))


# ### Map correlation function

  def mapCorrelation(self, im, x_im, y_im, vp, xs, ys):
    '''
    INPUT 
    im              the map 
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
      y1 = vp[1,:] + ys[jy] # 1 x 1076
      iy = np.int16(np.round((y1-ymin)/yresolution))
      for jx in range(0,nxs):
        x1 = vp[0,:] + xs[jx] # 1 x 1076
        ix = np.int16(np.round((x1-xmin)/xresolution))
        valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), np.logical_and((ix >=0), (ix < nx)))
        cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr


# ### Lidar to vehicle coordinate transform
# Convert from lidar frame to the vehicle frame

  def lidar_2_vehicle_frame(self, r, t):
      angles = np.linspace(-5, 185, 286) / 180 * np.pi

      # take valid indices
      indValid = np.logical_and((r < 80),(r> 0.1))
      r = r[indValid]
      angles = angles[indValid]

      
      # xy position in the sensor frame
      xs0 = r * np.cos(angles)
      ys0 = r * np.sin(angles)
      
      # convert position in the map frame here 
      R  = np.array([0.00130201, 0.796097, 0.605167,
                  0.999999, -0.000419027, -0.00160026,
                  -0.00102038, 0.605169, -0.796097 ]).reshape(3,3)
      p = np.array([0.8349, -0.0126869, 1.76416])

      # transformation matrix T
      T_vl = np.zeros((4,4))

      T_vl[:3,:3] = R
      T_vl[:3,-1] = p
      T_vl[-1,:] = np.array([0, 0, 0, 1])

      # create homogeneous coordinates in lidar frame
      lidar_cartesian = np.zeros((xs0.shape[0],4))

      lidar_cartesian[:,0] = xs0
      lidar_cartesian[:,1] = ys0
      lidar_cartesian[:,2] = 0
      lidar_cartesian[:,3] = 1

      # (A * B.T).T = B * A.T (Matrix property)
      # np.matmul(T_vl, lidar_cartesian.T).T = np.matmul(lidar_cartesian, T_vl.T)
      lidar_in_vehicle = np.matmul(lidar_cartesian, T_vl.T)

      x = lidar_in_vehicle[:,0]
      y = lidar_in_vehicle[:,1]
      z = lidar_in_vehicle[:,2]

      return np.array([x[z<=t], y[z<=t], z[z<=t]]).T


# ## Sigmoid function

  def sigmoid(self, a):
      gamma = 1 / ( 1 + np.exp(-a))
      return gamma


# ## Mapping

# ### Update logmap (map log odds)

  def map_log_odds(self, im, px, py, lx, ly, confidence):

      # convert from meters to cells
      pxi = np.ceil((px - im['xmin']) / im['res'] ).astype(np.int16) - 1
      pyi = np.ceil((py - im['ymin']) / im['res'] ).astype(np.int16) - 1 
      lxi = np.ceil((lx - im['xmin']) / im['res'] ).astype(np.int16) - 1
      lyi = np.ceil((ly - im['ymin']) / im['res'] ).astype(np.int16) - 1


      # map log odds using Bresenham 2D
      for k in range(lxi.shape[0]):
          empty_cells = self.bresenham2D(pxi, pyi, lxi[k], lyi[k]).astype(int)
          im['logmap'][empty_cells[0,:-1],empty_cells[1,:-1]] -= np.log(confidence)
          im['logmap'][empty_cells[0,-1],empty_cells[1,-1]] += np.log(confidence)
      
      return im['logmap']


# ### Update Bitmap (Bernoulli variables)

  def update_bitmap(self, im):
      im['bitmap'][im['logmap'] > 0] = 1 # obstacles
      im['bitmap'][im['logmap'] < 0] = 2 # free sapce
      im['bitmap'][im['logmap'] == 0] = 0 # unexplored

      return im['bitmap']
      


# ##  Particle Filter

if __name__ == '__main__':

  slam = PartFilterSLAM()

  x_im = np.arange(slam.MAP['xmin'],slam.MAP['xmax']+slam.MAP['res'],slam.MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(slam.MAP['ymin'],slam.MAP['ymax']+slam.MAP['res'],slam.MAP['res']) #y-positions of each pixel of the map

  # 9x9 array of pixels around a particle
  x_range = np.arange(-4, 4 + 1, 1)
  y_range = np.arange(-4, 4 + 1, 1)

  # Initialize the vehicle at (0,0) in world frame
  vehicle_in_world = [0,0]

  alpha = np.ones(slam.N_particles) / slam.N_particles      # particle weights
  part_state = np.zeros((slam.N_particles,3))          # initialize all particle states as (0,0,0)

  # ########################################  MAP INITIALIZE STEP 0 ######################
  # Read lidar file
  folder_path = os.path.dirname(os.path.abspath(__file__)) 
  _, lidar_data = slam.read_data_from_csv(os.path.join(folder_path, 'sensor_data/lidar.csv'))
  ranges = lidar_data[0, :]

  # Find (x,y) endpoints of lidar rays in vehicle frame 
  lidar_in_vehicle = slam.lidar_2_vehicle_frame(ranges, slam.z_thresh)

  # Update MAP log odds   
  slam.MAP['logmap'] = slam.map_log_odds(slam.MAP, vehicle_in_world[0], vehicle_in_world[1],
                              lidar_in_vehicle[:,0], lidar_in_vehicle[:,1], slam.lidar_confidence)
  slam.MAP['bitmap'] = slam.update_bitmap(slam.MAP)


  # ######################################  PREDICTION STEP ###########################
  # Load prediction values
  input_data = np.load(os.path.join(folder_path, 'input_data.npy'))
  del_t = input_data[:,0]                         # time difference b/w consecutive timestamps
  v = input_data[:,1]                             # linear velocity calc. from encoder data
  w = input_data[:,2]                             # angular velocity calculated from fog data 

  counter = 0                                     # no. of times prediction has happened
  trajectory = np.zeros((1,2))                    # array to store best particle trajectory at each update

  # Outer loop
  for i in tqdm(range(1,int(lidar_data.shape[0]/slam.ratio))):
      
      # Run 'ratio' prediction steps for each update step
      for j in range(slam.ratio * (i-1), slam.ratio * i):
          counter += 1 
          # Add gaussian noise to v and w with 0 mean
          v_noise = v[j] + np.random.normal(0, slam.v_var, slam.N_particles)
          w_noise = w[j] + np.random.normal(0, slam.w_var, slam.N_particles)
          
          # update particle states according to differential drive motion model
          part_state[:,0] += del_t[j] * v_noise * np.cos(part_state[:,2])
          part_state[:,1] += del_t[j] * v_noise * np.sin(part_state[:,2])
          part_state[:,2] += del_t[j] * w_noise                      

      # ######################################  UPDATE STEP ###############################

      # next lidar scan (after performing 'ratio' predictions)
      ranges = lidar_data[counter, :]
      c = np.zeros(slam.N_particles)                   # array to store correlation scores for each particle

      # Find (x,y) endpoints of lidar rays in vehicle frame 
      lidar_in_vehicle = slam.lidar_2_vehicle_frame(ranges, slam.z_thresh)
      # Make 4D coordinate by appending ones
      lidar_in_vehicle =  np.append(lidar_in_vehicle, np.ones((lidar_in_vehicle.shape[0],1)),axis = 1)

      # obtain correlation for each of the N particles
      for j in range(slam.N_particles):

          # Pose of particle in world frame
          T_wp = np.array([np.cos(part_state[j,2]), -np.sin(part_state[j,2]), 0, part_state[j,0], 
                          np.sin(part_state[j,2]), np.cos(part_state[j,2]), 0, part_state[j,1],
                      0, 0, 1, 0,
                      0, 0, 0, 1]).reshape(4,4)


          # (A * B.T).T = B * A.T (Matrix property of transpose distribution)
          # np.matmul(T_wp, lidar_cartesian.T).T = np.matmul(lidar_cartesian, T_wp.T)
          # lidar endpoints in world coordinates according to particle j
          lidar_in_world = np.matmul(lidar_in_vehicle, T_wp.T)        

          # Map correlation    
          Y = np.stack((lidar_in_world[:,0],lidar_in_world[:,1]))
          # taking average of all the correlation values in the 9x9 array
          c[j] = np.average(slam.mapCorrelation(slam.MAP['bitmap'],x_im,y_im,Y,x_range,y_range))

      
      # scale alphas and renormalize
      if np.any(c == 0):                          # if c ==0, scaling would be make the corressponding particle obsolete, so resample 
          resample_idx = np.random.choice(slam.N_particles,size = slam.N_particles, p = alpha)
          part_state = part_state[resample_idx]
          alpha = np.ones(slam.N_particles) / slam.N_particles
      else:
          alpha = alpha * c / np.sum(alpha * c)
      
      # Pick particle with highest score
      part_idx = np.argmax(c)

      # trajectory of best particle
      trajectory = np.append(trajectory, np.array([part_state[part_idx,0], part_state[part_idx,1]]).reshape(1,2), axis = 0)

      # Pose of particle in world frame
      T_wp = np.array([np.cos(part_state[part_idx,2]), -np.sin(part_state[part_idx,2]), 0, part_state[part_idx,0], 
                      np.sin(part_state[part_idx,2]), np.cos(part_state[part_idx,2]), 0, part_state[part_idx,1],
                  0, 0, 1, 0,
                  0, 0, 0, 1]).reshape(4,4)


      # (A * B.T).T = B * A.T (Matrix property)
      # np.matmul(T_vl, lidar_cartesian.T).T = np.matmul(lidar_cartesian, T_vl.T)
      lidar_in_world = np.matmul(lidar_in_vehicle, T_wp.T)

      # Update map log odds
      slam.MAP['logmap'] = slam.map_log_odds(slam.MAP, part_state[part_idx,0], part_state[part_idx,1],
                                  lidar_in_world[:,0], lidar_in_world[:,1], slam.lidar_confidence)
      slam.MAP['bitmap'] = slam.update_bitmap(slam.MAP)

      # Resampling
      N_eff = np.sum(1/(alpha * alpha))

      if N_eff < slam.resample_thresh:
          resample_idx = np.random.choice(slam.N_particles,size = slam.N_particles, p = alpha)
          part_state = part_state[resample_idx]
          alpha = np.ones(slam.N_particles) / slam.N_particles
  
  
  # Plotting all the required figures
  
  # Load dead-reckoning coordinates
  dead_reckon = slam.robot_path()

  # Find (x,y) endpoints of lidar rays for first lidar scan 
  slam_2 = PartFilterSLAM()
  ranges = lidar_data[0, :]
  lidar_in_vehicle = slam_2.lidar_2_vehicle_frame(ranges, 1E10) # large value so that thresholding is not active, else very few rays will be plotted 
  slam_2.MAP['logmap'] = slam_2.map_log_odds(slam_2.MAP, vehicle_in_world[0], vehicle_in_world[1],
                              lidar_in_vehicle[:,0], lidar_in_vehicle[:,1], slam_2.lidar_confidence)
  slam_2.MAP['bitmap'] = slam_2.update_bitmap(slam_2.MAP)
  
  fig1, (ax1) = plt.subplots(1, 1, figsize = (10,15))
  ax1.imshow(slam_2.MAP['bitmap'], cmap = 'hot');
  ax1.set_title('LIDAR Scan Visualization')
  ax1.set_xlim([1450,1600])
  ax1.set_ylim([1600,1450])

  fig2, (ax2) = plt.subplots(1, 1, figsize = (10,15))
  ax2.plot(dead_reckon[:,0], dead_reckon[:,1]);
  ax2.set_xlabel('x - axis displacement')
  ax2.set_ylabel('y - axis displacement')
  ax2.set_title('Robot path - dead reckoning')

  fig3, (ax3) = plt.subplots(1, 1, figsize = (10,15))
  ax3.imshow(slam.MAP['bitmap'], cmap = 'hot');
  ax3.set_title('Occupancy grid map for \n N_particles = %d, Predict Frequency = %d,\n v_noise = %.1E, w_noise = %.1E' %(slam.N_particles, slam.ratio, slam.v_var, slam.w_var))
  ax3.set_xlabel('x - dimensions')
  ax3.set_ylabel('y - dimensions')
  ax3.set_xlim([0,2000])
  ax3.set_ylim([3000,1000])

  plt.tight_layout()
  plt.show(block = True)

