# # Project 3 Prediction

# ## Import packages
import numpy as np
import matplotlib.pyplot as plt
from pr3_utils import *
import os, math
from tqdm import tqdm
from scipy import linalg
import matplotlib
#matplotlib.use('Qt5Agg')

class ViSLAM():
    
    
    def __init__(self):
        """ Load all required measurements and also initialize all required variables
        """

        # ## Load measurements
        folder_path = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(folder_path, 'data/03.npz') 
        t,features_complete,self.linear_velocity,self.angular_velocity,\
            self.K,self.b,self.imu_T_cam = load_data(self.filename)

        # skip features to reduce runtime
        skip = 10 if self.filename[-6:-4] == '03' else 50
        self.features = features_complete[:,::skip,:]

        # remove timesteps where no features are observed (all entries -1)
        # ~ 30 steps in 1000 timesteps for 03.npz
        todelete = np.argwhere(np.all(self.features[0,:,:] == -1, axis = 0))
        todelete = np.ndarray.flatten(todelete)
        t =  np.delete(t, todelete, 1)
        self.features =  np.delete(self.features, todelete, 2)
        self.linear_velocity =  np.delete(self.linear_velocity, todelete, 1)
        self.angular_velocity =  np.delete(self.angular_velocity, todelete, 1)
        self.del_t = np.diff(t)
        
        # IMU Parameters 
        self.mu_IMU = np.eye(4)                                           
        self.sigma_IMU = 1e-5 * np.eye(6)                                        
        self.W = 1e-5 * np.eye(6)                                                          
        self.T = np.eye(4).reshape(4,4,1)
        # Mapping parameters
        self.M = self.features.shape[1]         # no. of features
        self.mu_MAP = np.zeros((self.M,3))
        self.sigma_MAP = 1e-5 * np.eye(3 * self.M)
        self.Ks = np.row_stack((self.K[:-1,:], self.K[0,:], self.K[1,:]))   #camera matrix
        self.Ks = np.column_stack((self.Ks,[0, 0, -self.K[0,0]*self.b, 0])) 
        self.P = np.column_stack((np.eye(3), np.zeros((3,1))))
        self.V = 1e-3

        # reotating camera transformation as was suggested
        R_x = np.array([1, 0, 0, 0, 
                        0, -1, 0, 0, 
                        0, 0, -1, 0, 
                        0, 0, 0, 1]).reshape(4,4)
        self.imu_T_cam = R_x @ self.imu_T_cam

        # concatenate both sigmas to update in one single step
        self.sigma = linalg.block_diag(self.sigma_MAP, self.sigma_IMU)



    # ## Define hatmap & curly hatmap & transformation matrices
    def se3_2_SE3(self,v,w):    #4x4
        arr = [ 0,     -w[2],   w[1],   v[0],
                w[2],   0,     -w[0],   v[1],
                -w[1],   w[0],   0,      v[2],
                0,       0,      0,      0    ]
        transform = np.array(arr).reshape(4,4)
        return transform

    def hatmap(self,w):         #3x3
        arr = [  0,     -w[2],   w[1],
                w[2],   0,     -w[0],
                -w[1],   w[0],   0,   ]
        arr = np.array(arr).reshape(3,3)
        return arr
        
    def se3_2_6x6(self,v,w):    #6x6
        arr = [ 0,     -w[2],   w[1],   0,     -v[2],   v[1],
                w[2],   0,     -w[0],   v[2],   0,     -v[0],
                -w[1],   w[0],   0,     -v[1],   v[0],   0,
                0,       0,      0,     0,     -w[2],   w[1],
                0,       0,      0,     w[2],   0,     -w[0],
                0,       0,      0,     -w[1],   w[0],   0,     
                ]
        curly_hatmap = np.array(arr).reshape(6,6)
        return curly_hatmap


    # ## pi function & dpi/dq; q is a 4x1 vector
    def pi_func(self,q):
        return q / q[:,2].reshape(q.shape[0],1)

    def diff_pi_func(self,q): #4x1 
        arr = np.array([    1, 0, -q[0]/q[2], 0,
                            0, 1, -q[1]/q[2], 0,
                            0, 0,  0,         0,
                            0, 0, -q[3]/q[2], 1]).reshape(4,4)
        return arr


# ## Extended Kalman Filter


if __name__ == '__main__':
    
    slam = ViSLAM()

    del_mu_IMU = np.random.normal(0,np.diag(slam.sigma_IMU))                                  
    del_mu_hat_IMU = slam.se3_2_SE3(del_mu_IMU[:3], del_mu_IMU[3:])   
    slam.T = (slam.mu_IMU @ linalg.expm(del_mu_hat_IMU)).reshape(4,4,1)    
    
    
    for i in tqdm(range(slam.del_t.shape[1])): #del_t.shape[1]
        ######### IMU PREDICTION BEGINS ###################
        # Calculating terms to be used for prediction step
        exp_tu_hat = linalg.expm(slam.del_t[0,i] * slam.se3_2_SE3(\
            slam.linear_velocity[:,i], slam.angular_velocity[:,i]))
        
        exp_tu_adj = linalg.expm(-slam.del_t[0,i] * slam.se3_2_6x6(\
            slam.linear_velocity[:,i], slam.angular_velocity[:,i]))

        # IMU prediction step
        slam.mu_IMU = slam.mu_IMU @ exp_tu_hat
        slam.sigma[-6:,-6:] = exp_tu_adj @ slam.sigma[-6:,-6:] @ exp_tu_adj.T + slam.W
        ######### IMU PREDICTION ENDS ###################
    

        ## Calculate pose using perturbation theory
        del_mu_IMU =  exp_tu_adj @ del_mu_IMU + np.random.normal(0,np.diag(slam.W))
        del_mu_hat_IMU = slam.se3_2_SE3(del_mu_IMU[:3], del_mu_IMU[3:])
        slam.T = np.append(slam.T, (slam.mu_IMU @ linalg.expm(del_mu_hat_IMU)).reshape(4,4,1), axis = 2)



    visualize_trajectory_2d(slam.T, 'Path using %s' %slam.filename[-7:], True)