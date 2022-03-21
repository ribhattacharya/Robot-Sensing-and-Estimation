'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os
import cv2
from skimage.measure import label, regionprops

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		#Setting file names from folder
		folder_path = os.path.dirname(os.path.abspath(__file__)) 
		file_theta = os.path.join(folder_path, 'trained_theta.npy')
		file_mu = os.path.join(folder_path, 'trained_mu.npy')
		file_sigma = os.path.join(folder_path, 'trained_sigma.npy')

		#loading files into the parameters
		self.theta = np.load(file_theta)
		self.mu = np.load(file_mu)
		self.sigma = np.load(file_sigma)

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		#load image as divided by 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255
		
		
		#no. of labels {1, 2, 3, 4} and no. of features {R, G, B}
		k , d = self.theta.shape[0] , img.shape[2]

		#Convert 3D array to 2D array for better comprehension 
        # and vectorization
		r, c = img.shape[0], img.shape[1]

		#flatten the image while keeping total size intact
		flat_img = np.reshape(img, (r*c,img.shape[2]))
		mask_img = np.zeros_like(flat_img)
		pxy = np.zeros((r*c,k))

		#Find p(x|y) for each data point and label;
		for l in range(k):
			pxy[:,l] = 1/np.sqrt(np.abs(((2*np.pi)**d) * np.linalg.det(self.sigma[l,:,:]))) * \
				np.exp(-0.5 * (  np.sum((flat_img - self.mu[l,:])* np.matmul( 
					np.linalg.inv(self.sigma[l,:,:]),  (flat_img - self.mu[l,:]).T).T, 
										axis = 1) 
								)
						)
			
		#Find the argmax of the single pixel
		y = np.argmax(pxy * self.theta.T, axis = 1) + 1
		
        #create a boolean mask for blue regions (y==1)
		mask_img = y==1

		#reshape mask to match the dimensions of the input image
		mask_img = np.reshape(mask_img, (r,c))
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		# mask_img = self.segment_image(img)

		
		#create image labels and regions from masked image
		labels = label(img)
		props = regionprops(labels)

		#initialize array to be returned
		boxes = []

		#loop over all regions detected in the mask
		for region in props:

            #get coordinates of the bounding box
            #upon inspection and debugging, this method was giving 
            # me bbox outputs in the form (y1, x1, y2, x2) instead of the usual 
            # (x1, y1, x2, y2). That is why the height and width have been taken accordingly.

			box = region.bbox
			height = abs(box[0] - box[2])
			width = abs(box[1] - box[3])


            # shape statistic, as has been explained in the report
			if (height/width) >= 1 and (height/width) <= 2 and (region.area > 0.002 * img.shape[0] * img.shape[1]) :
				
				
				#return the bbox coordinates by 
                # transforming from (y1, x1, y2, x2) to (x1, y1, x2, y2) 
				boxes.append([box[1], box[0], box[3], box[2]])
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes


