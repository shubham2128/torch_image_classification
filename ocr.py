import cv2
import os
import numpy as np

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

# Function to create the '/ocr' directory if it doesn't exist
def create_directory():
    if not os.path.exists('ocr'):
        os.makedirs('ocr')

# Load an image
image_path = '/Users/shubham/Documents/vinove/dataset/1000_F_104356820_2CoImRpN4PPXotsEcD8CRnnrqj4ykhZd.jpg'
image = load_image(image_path)

if image is not None:
    # Create the '/ocr' directory
    create_directory()

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_image = cv2.magnitude(sobel_x, sobel_y)
    cv2.imwrite('ocr/edge_image.jpg', edge_image.astype(np.uint8))

    # Apply thresholding
    ret, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('ocr/binary_image.jpg', binary_image)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    segmented_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('ocr/segmented_image.jpg', segmented_image)

    print("Processed images saved successfully in the 'ocr' folder.")
