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

# Function to create the '/feature_extraction' directory if it doesn't exist
def create_directory():
    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')

# Load an image
image_path = 'dataset/car_0285.jpg'
image = load_image(image_path)

if image is not None:
    # Create the '/feature_extraction' directory
    create_directory()

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('feature_extraction/gray_image.jpg', gray_image)

    # Perform Fourier Transform
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
    cv2.imwrite('feature_extraction/magnitude_spectrum.jpg', magnitude_spectrum.astype(np.uint8))

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite('feature_extraction/image_with_keypoints.jpg', image_with_keypoints)

    # Define the region to be inpainted
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    mask[100:300, 100:300] = 255
    cv2.imwrite('feature_extraction/mask.jpg', mask)

    # Perform inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite('feature_extraction/inpainted_image.jpg', inpainted_image)

    print("Modified images saved successfully in the '/feature_extraction' folder.")
