import cv2
import numpy as np

# Function to process an image: apply Canny edge detection first, then dilation
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Apply bilateral filter for noise reduction
    noise_reduced_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(noise_reduced_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Threshold the image - this value may need adjustment for your images
    ret, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours from the binary image
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image for contours which is the same size as the original image
    contour_image = np.zeros_like(thresholded_image)

    # Draw the contours on the contour image
    cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)

    # Perform morphological operations to further clean up the image
    kernel = np.ones((5, 5), np.uint8)
    contour_image = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel, iterations=4)  # Remove noise
    dilated_image = cv2.dilate(contour_image, kernel, iterations=5)  # Fill in the gaps

    final_image = cv2.bitwise_not(dilated_image)

    return final_image

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