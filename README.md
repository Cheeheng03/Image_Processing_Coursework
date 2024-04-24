# Image Processing Pipeline

This repository contains Python code for processing images using computer vision techniques, primarily focusing on segmenting specific objects from images. 
The pipeline includes functions for preprocessing images, identifying contours, calculating Intersection over Union (IoU) metrics for evaluation, and displaying visual results.

## Requirements
1) Python 3.x 
2) OpenCV (cv2)
3) NumPy

## Installation
1) Ensure you have Python installed on your system. If not, you can download it from <a href="hhttps://www.python.org" target="_blank">Python's official website</a>.
2) Install the required dependencies using pip:
```commandline
pip install opencv-python numpy
```

## Usage
1) Clone or download this repository to your local machine.
2) Ensure that the original flower inout images are in the Dataset/input_images directory. 
Ensure that the ground truth images are placed in corresponding directories under Dataset/ground_truths.
3) Run Main.py to get the result of each image compared with the ground truth images.
4) Check the similarities' percentage for each pair of processed image and ground truth image.

## Codes
### Functions
1) process_image(image_path, output_name): Processes an image using a predefined pipeline, including noise reduction, thresholding, contour detection, and morphological operations. 
Saves the processed image to the specified output directory.
2) process_red_edges(image): Processes the ground truth image specifically targeting red-colored edges for segmentation.
3) find_all_contours(binary_image): Finds all contours in a binary image.
4) calculate_iou(maskA, maskB): Calculates the Intersection over Union (IoU) score between two masks.
5) display_similarity(image_name, iou_score): Displays the similarity percentage based on the IoU score.
6) show_binary_image(image, window_name='Binary Image'): Displays a binary image.
7) combine_images_side_by_side(images, labels, output_name): Display and label the processed image mask, ground truth mask and overlay image side by side.

### Flow
1) Image Preparation: Extract the filename of the input image, then read the original image for visualization purposes using OpenCV's cv2.imread() function.
2) Image Processing: Call the process_image() function to process the input image. This function applies a series of operations, including noise reduction, thresholding, contour detection, and morphological operations, returning the processed image.
3) Ground Truth Image Handling: Read the ground truth image using cv2.imread(). If the dimensions of the processed image and the ground truth image do not match, resize the ground truth image to match the dimensions of the processed image.
4) Contour Detection: Utilize the find_all_contours() function to detect all contours in the processed image. Additionally, apply the process_red_edges() function to find contours in the ground truth image.
5) Mask Creation: Create binary masks for both the processed image and the ground truth image by drawing contours on blank images using cv2.drawContours(). This step is crucial for calculating the Intersection over Union (IoU) later.
6) IoU Calculation: Calculate the IoU score between the processed image mask and the ground truth mask using the calculate_iou() function. This metric evaluates the similarity between the processed image and the ground truth.
7) Visualization: Display the processed image mask, ground truth image mask, and an overlay of both images side by side with labels using combine_images_side_by_side function. The overlay image highlights the detected contours, with ground truth contours in green and processed contours in red.
8) IoU Evaluation: Append the calculated IoU score to a list (ious) for further analysis.
9) Average Similarities' Percentage: Calculate the sum of the similarities' percentage and average out and print out the average similarities' percentage for all the 9 images compared with their respective ground truth.
10) Cleanup: Close all OpenCV windows using cv2.destroyAllWindows().

## Output
* The processed images will be saved in the processed_images directory.
* Visualizations of the processed image mask, ground truth image mask, and overlay will be displayed for each processed image.
* Similarities' Percentages for the pair of processed image and ground truth image will be shown.
* The average similarity's percentage will be calculated and shown.