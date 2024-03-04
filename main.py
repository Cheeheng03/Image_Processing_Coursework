import cv2
import numpy as np


# Function to process an image: reduce noise and convert color space
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Apply bilateral filter for noise reduction
    # Parameters: src, d, sigmaColor, sigmaSpace
    noise_reduced_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to a lower-dimensional color space (grayscale)
    grayscale_image = cv2.cvtColor(noise_reduced_image, cv2.COLOR_BGR2GRAY)

    return grayscale_image


# Paths to your flower images
image_paths = ['Dataset/input_images/easy/easy_1.jpg', 'Dataset/input_images/easy/easy_2.jpg', 'Dataset/input_images/easy/easy_3.jpg',
               'Dataset/input_images/medium/medium_1.jpg', 'Dataset/input_images/medium/medium_2.jpg', 'Dataset/input_images/medium/medium_3.jpg',
               'Dataset/input_images/hard/hard_1.jpg', 'Dataset/input_images/hard/hard_2.jpg', 'Dataset/input_images/hard/hard_3.jpg']

# Process each image
processed_images = [process_image(path) for path in image_paths]

# Display each processed image one by one
for processed_image in processed_images:
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
