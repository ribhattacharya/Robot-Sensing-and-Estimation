import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    # read the folder
    root_folder = os.path.dirname(os.path.abspath(__file__)) 
    folder = os.path.join(root_folder, 'data/training')

    # loop over the entire working directory one file at a time
    for filename in sorted(os.listdir(folder)): 
  
        cont = 'y'      # user defined delimiter, which breaks from the loop if cont != 'y'

        #parse over all .jpg files
        #while loop is used to revisit the same image to classify another region/label
        while cont == 'y' and filename.endswith('.jpg'):
            
            #read the image from working directory
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # display the image and use roipoly for labeling
            fig, ax = plt.subplots()
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
            
            # get the image mask
            mask = my_roi.get_mask(img)
            
            # display the labeled region and the image mask
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
            
            ax1.imshow(img)
            ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
            ax2.imshow(mask)
            
            
            plt.show(block=True)
            
            #accept label input from the user for the masked pixels
            label = int(input("Which label is this? {blue bin = 1, blue not bin = 2, other bins = 3, everything else = 4}:\t"))
            
            #store masked pixels and their labels in data
            X = img[mask==1]
            y = np.full(X.shape[0],label).T
            data = np.concatenate((X,y[:, None]), axis = 1)
            
            #waiting for user input to revisit the image 
            # for collecting another positive/negative example
            cont = input("Do you want to use the same image again? (y/n) (Type 'exit' to quit):\t")
            
            #save the data to a file, which is specific to an image and label combination
            #This is done to introduce fragmentation, which can be useful in case some specific 
            # training data needs to be removed, without affecting other pixel values
            new_file = os.path.join(root_folder, 'collected_data', filename[:-4]+'_%d.npy' % label)
            if os.path.exists(new_file):
                temp = np.load(new_file)
                data = np.unique(np.append(temp, data, axis = 0), axis = 0)

            #Save the data for each particular image and label
            np.save(new_file, data)
        
        #break from the outer loop in case no 
        # more data needs to be collected
        if cont == 'exit':
            break
    
 
    #Collate all data into single file, 
    # while keeping old contents intact
    data = np.empty((0,4))
    folder = os.path.join(root_folder, 'collected_data')
    for filename in sorted(os.listdir(folder)):
            if filename.endswith('.npy'):
                data = np.unique(np.append(
                    data, np.load(os.path.join(folder, filename)), axis = 0), axis = 0)
    np.save(os.path.join(folder, 'complete_data.npy'), data)



