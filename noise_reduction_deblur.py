import cv2
import numpy as np
import os

# Function to load an image with error handling
def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Error: Image not found or cannot be read.")
        return image
    except Exception as e:
        print(str(e))
        return None

# Function to create the 'noise_reduction_deblur' directory if it doesn't exist
def create_directory():
    if not os.path.exists('noise_reduction_deblur'):
        os.makedirs('noise_reduction_deblur')

# Load an image
image_path = '/Users/shubham/Documents/vinove/dataset/car_0278.jpg'
image = load_image(image_path)

if image is not None:
    # Create the 'noise_reduction_deblur' directory
    create_directory()

    # 1. Noise Reduction
    # Apply Gaussian blur
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite('noise_reduction_deblur/denoised_image.jpg', denoised_image)

    # Apply median filtering
    median_filtered_image = cv2.medianBlur(image, 5)
    cv2.imwrite('noise_reduction_deblur/median_filtered_image.jpg', median_filtered_image)

    # 2. Deblurring (Wiener)
    # Define the Wiener filter kernel
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    # Apply Wiener deblurring using filter2D
    deblurred_image_wiener = cv2.filter2D(image, -1, kernel)
    cv2.imwrite('noise_reduction_deblur/deblurred_image_wiener.jpg', deblurred_image_wiener)

    print("Modified images saved successfully in the 'noise_reduction_deblur' directory.")
