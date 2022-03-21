import yaml, cv2, os
import numpy as np
from skimage.measure import label, regionprops
from skimage.draw import rectangle_perimeter
from skimage.morphology import erosion, dilation
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#Setting file names from folder
root_folder = os.path.dirname(os.path.abspath(__file__)) 
file_theta = os.path.join(root_folder, 'trained_theta.npy')
file_mu = os.path.join(root_folder, 'trained_mu.npy')
file_sigma = os.path.join(root_folder, 'trained_sigma.npy')

#loading files into the parameters
theta = np.load(file_theta)
mu = np.load(file_mu)
sigma = np.load(file_sigma)

folder  = os.path.join(root_folder, "data/validation")

for filename in sorted(os.listdir(folder)):    

    # read one test image
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #no. of labels {1, 2, 3} and no. of features {R, G, B}
        k , d = theta.shape[0] , img.shape[2]

        #Convert 3D array to 2D array for better comprehension 
        # and vectorization
        r, c = img.shape[0], img.shape[1]
        
        #flatten the image while keeping total size intact
        flat_img = np.reshape(img, (r*c,img.shape[2])).astype(np.float64)/255
        mask_img = np.zeros_like(flat_img)
        pxy = np.zeros((r*c,k))

        #Find p(x|y) for each data point and label;
        for l in range(k):
            pxy[:,l] = 1/np.sqrt(np.abs(((2*np.pi)**d) * np.linalg.det(sigma[l,:,:]))) * \
                np.exp(-0.5 * (  np.sum((flat_img - mu[l,:])* np.matmul( 
                    np.linalg.inv(sigma[l,:,:]),  (flat_img - mu[l,:]).T).T, 
                                        axis = 1) 
                                )
                        )
            
        #Find the argmax of the single pixel
        y = np.argmax(pxy * theta.T, axis = 1) + 1
        
        #create a boolean mask for blue regions (y==1)
        mask_img = (y==1)

        #reshape mask to match the dimensions of the input image
        mask_img = np.reshape(mask_img, (r,c))


        #create a figure to collect and compare the images
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,15))

        #create image labels and regions from masked image
        new = label(mask_img)
        props = regionprops(new)
        
        ##Show input image, mask and labeled image
        ax1.imshow(img)
        ax2.imshow(mask_img)
        ax2.set_title('Mask using GDA classifier')
        ax3.imshow(new)
        ax3.set_title('Labelled mask using skimage.label')
        

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
            if float(height/width) >= 1 and float(height/width) <= 2 and (region.area > 0.02 * img.shape[0] * img.shape[1]) :
                
                #If box passes shape statistic filter, 
                # the draw the box
                start = (box[0],box[1])
                end = (box[2],box[3])
                row, col = rectangle_perimeter(start, end)

                #show the bbox coordinates by 
                # transforming from (y1, x1, y2, x2) to (x1, y1, x2, y2) 
                ax1.set_title('bbox: ' + str([box[1], box[0], box[3], box[2]]))
                ax1.plot(col, row)
        plt.show()
    cv2.destroyAllWindows()



