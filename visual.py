import os
import cv2
import numpy as np

folder_path = '/Users/shubham/Documents/vinove/dataset'
display_limit = 5  # Limit the number of images to display
counter = 0
images = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            
            # Display the image using OpenCV
            cv2.imshow('Image', img)
            cv2.waitKey(0)  # Wait for a key press to close the image window
            cv2.destroyAllWindows()  # Close the image window
            # Increment the counter
            counter += 1       
            # Break the loop if the limit is reached
            if counter >= display_limit:
                break
