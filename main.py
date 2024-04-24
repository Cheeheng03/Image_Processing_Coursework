import cv2
import numpy as np
import os

# Function to process an image using the same pipeline as segment_flowers
def process_image(image_path, output_name):
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
    contour_image = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise
    dilated_image = cv2.dilate(contour_image, kernel, iterations=3)  # Fill in the gaps

    binary_image = cv2.bitwise_not(dilated_image)

    # Find all contours in the processed image
    contours_processed = find_all_contours(binary_image)

    # Combine all ground truth contours into a single mask
    final_image = np.zeros(binary_image.shape, dtype=np.uint8)
    for contour in contours_processed:
        cv2.drawContours(final_image, [contour], -1, 255, thickness=cv2.FILLED)

    # Write processed image to the directory
    cv2.imwrite(os.path.join(output_dir, output_name), final_image)

    return final_image

def process_red_edges(image):
    # Smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Adjusted HSV range for red color
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological closing to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Otsu thresholding
    _, edges = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# Function to find all contours in the image
def find_all_contours(binary_image):
    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_iou(maskA, maskB):
    intersection = np.logical_and(maskA, maskB)
    union = np.logical_or(maskA, maskB)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Function to display similarity percentage based on IoU
def display_similarity(image_name, iou_score):
    similarity_percentage = round(iou_score * 100, 2)
    print(f"Similarity for {image_name}: {similarity_percentage}%")

def show_binary_image(image, window_name='Binary Image'):
    # Display the binary image
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def combine_images_side_by_side(images, labels, output_name):
    # Calculate the dimensions for the combined image
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + (10 * (len(images) - 1))
    combined_image = np.ones((height + 50, width, 3), dtype=np.uint8) * 255  # +50 for label space

    # Place each image side by side
    x_offset = 0
    for idx, image in enumerate(images):
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        combined_image[50:50+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        cv2.putText(combined_image, labels[idx], (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        x_offset += image.shape[1] + 10

    cv2.imshow(f"Combined: {output_name}", combined_image)
    cv2.waitKey(0)


# Make sure the output directory exists
output_dir = 'processed_images'
os.makedirs(output_dir, exist_ok=True)

# Paths to your images
image_paths = ['Dataset/input_images/easy/easy_1.jpg', 'Dataset/input_images/easy/easy_2.jpg',
               'Dataset/input_images/easy/easy_3.jpg',
               'Dataset/input_images/medium/medium_1.jpg', 'Dataset/input_images/medium/medium_2.jpg',
               'Dataset/input_images/medium/medium_3.jpg',
               'Dataset/input_images/hard/hard_1.jpg', 'Dataset/input_images/hard/hard_2.jpg',
               'Dataset/input_images/hard/hard_3.jpg']

ground_truth_image_paths = ['Dataset/ground_truths/easy/easy_1.png', 'Dataset/ground_truths/easy/easy_2.png',
                            'Dataset/ground_truths/easy/easy_3.png',
                            'Dataset/ground_truths/medium/medium_1.png', 'Dataset/ground_truths/medium/medium_2.png',
                            'Dataset/ground_truths/medium/medium_3.png',
                            'Dataset/ground_truths/hard/hard_1.png', 'Dataset/ground_truths/hard/hard_2.png',
                            'Dataset/ground_truths/hard/hard_3.png']

# Apply the processing and calculate IoU for each image
ious = []
for input_path, ground_truth_path in zip(image_paths, ground_truth_image_paths):
    image_name = os.path.basename(input_path)
    original_image = cv2.imread(input_path)  # Read the original image again for visualization
    processed_image = process_image(input_path, image_name)
    ground_truth_image = cv2.imread(ground_truth_path)

    if processed_image.shape != ground_truth_image.shape:
        ground_truth_image = cv2.resize(ground_truth_image, (processed_image.shape[1], processed_image.shape[0]))

    # Find all contours in the processed image
    contours_processed_image = find_all_contours(processed_image)

    # Find all contours in the ground truth image
    contours_ground_truth = process_red_edges(ground_truth_image)

    # Combine all ground truth contours into a single mask
    ground_truth_mask = np.zeros(processed_image.shape, dtype=np.uint8)
    for contour in contours_ground_truth:
        cv2.drawContours(ground_truth_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Calculate IoU using the combined mask
    iou_score = calculate_iou(processed_image, ground_truth_mask)
    display_similarity(image_name, iou_score)

    ious.append(iou_score)

    # Create overlay images for visualization
    overlay_image = original_image.copy()

    # Draw ground truth contours in green
    for contour_processed in contours_processed_image:
        cv2.drawContours(overlay_image, [contour_processed], -1, (0, 0, 255), 3)

    # Draw ground truth contours in green
    for contour_ground_truth in contours_ground_truth:
        cv2.drawContours(overlay_image, [contour_ground_truth], -1, (0, 255, 0), 3)

    labels = ['Processed Mask', 'Ground Truth Mask', 'Overlay']
    images = [processed_image, ground_truth_mask, overlay_image]

    # Use basename of the input path to create a unique output name for each combined image
    output_name = f"combined_{os.path.basename(input_path).split('.')[0]}.jpg"

    combine_images_side_by_side(images, labels, output_name)


    cv2.destroyAllWindows()

average_iou_score = sum(ious) / len(ious) if ious else 0
print(f"Average IoU Score: {average_iou_score * 100:.2f}%")