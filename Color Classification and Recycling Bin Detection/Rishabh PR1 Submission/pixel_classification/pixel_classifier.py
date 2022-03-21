'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import os
import numpy as np

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    #Setting file names from working folder
    folder_path = os.path.dirname(os.path.abspath(__file__)) 
    file_theta = os.path.join(folder_path, 'trained_theta.npy')
    file_mu = os.path.join(folder_path, 'trained_mu.npy')
    file_sigma = os.path.join(folder_path, 'trained_sigma.npy')

    #loading files into the parameter variables
    self.theta = np.load(file_theta)
    self.mu = np.load(file_mu)
    self.sigma = np.load(file_sigma)
    
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
  
    # no. of labels {1, 2, 3} and no. of features {R, G, B}, 
    # which can be determined from the shapes of our input and parameters
    k , d = self.theta.shape[0] , X.shape[1]

    #initializing the conditional pdf
    pxy = np.zeros([X.shape[0],k])
     
    #Find p(x|y) for each label;
    for l in range(k):
      pxy[:,l] = 1/np.sqrt(np.abs(((2*np.pi)**d) * np.linalg.det(self.sigma[l,:,:]))) * \
            np.exp(-0.5 * (  np.sum((X-self.mu[l,:])* np.matmul( 
              np.linalg.inv(self.sigma[l,:,:]),  (X - self.mu[l,:]).T).T, axis = 1) 
            ))

    #Find the argmax of the product, 
    # add 1 since python indexes begin from 0
    y = np.argmax(pxy * self.theta.T, axis=1) + 1 

    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

