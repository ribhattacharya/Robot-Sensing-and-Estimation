'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import os


def train(data):
  '''
    Train a set of pixels into red, green, or blue
    
    Inputs:
      X: n x 3 matrix of RGB values
      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    Outputs:
      Parameters theta, mu and sigma
  '''
  ################################################################
  # YOUR CODE AFTER THIS LINE
  X = data[:,:3]                         #training data
  y = data[:,3]                          #training labels
  n = len(y)                          #Total training set data points
  d = X.shape[1]                      #no. of features, {R, G, B} 
  k = np.count_nonzero(np.unique(y))  #no. of labels, y = {1, 2, 3}

  #Initialize the parameters for Gaussian Discriminant Analysis
  theta = np.zeros([k,1])
  mu = np.zeros([k,d])
  sigma = np.zeros([k,d,d])

  #Compute and store the parameters for each 'y' label
  i = 0
  for label in np.unique(y):
    X_k = X[y==label,:]         #Observations for each label y = {1, 2, 3}

    theta[i,:] = np.count_nonzero(y==label)/n
    mu[i,:] = np.mean(X_k, axis=0)
    sigma[i,:,:] = np.cov(X_k, rowvar=False, bias=False)
    i+=1

  return theta, mu, sigma



if __name__ == '__main__':
  #Setting file names from working folder
  root_folder = os.path.dirname(os.path.abspath(__file__)) 
  training_dataset = os.path.join(root_folder, 'trained_X_y.npy')

  #load the dataset into an array
  data = np.load(training_dataset)
  
  #implement training to calculate and store the parameters
  theta, mu, sigma = train(data)

  #saving all parameters in dedicated .npy files each
  file_theta = os.path.join(root_folder, 'trained_theta.npy')
  file_mu = os.path.join(root_folder, 'trained_mu.npy')
  file_sigma = os.path.join(root_folder, 'trained_sigma.npy')

  np.save(file_theta, theta)
  np.save(file_mu, mu)
  np.save(file_sigma, sigma)