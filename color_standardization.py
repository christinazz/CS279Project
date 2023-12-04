import cv2
import numpy as np
import os

# Function to standardize colors across all images based on a given reference image
def color_standardization(image, reference_image):
    # Convert images to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    # Create a mask for black pixels
    black_mask = (image_lab[:, :, 0] == 0) & (image_lab[:, :, 1] == 128) & (image_lab[:, :, 2] == 128)

    # Calculate mean and standard deviation for each channel in LAB color space
    mean_image, stddev_image = cv2.meanStdDev(image_lab, mask=~black_mask.astype(np.uint8))
    mean_reference, stddev_reference = cv2.meanStdDev(reference_lab)

    # Perform color standardization
    standardized_image = image_lab.copy()

    for i in range(3):  # Iterate over LAB channels
        if i == 0:  # L channel (brightness) remains unchanged
            continue

        standardized_image[:, :, i] = ((image_lab[:, :, i] - mean_image[i]) * (stddev_reference[i] / stddev_image[i])) + mean_reference[i]

    # Convert back to BGR color space
    standardized_image = cv2.cvtColor(standardized_image, cv2.COLOR_LAB2BGR)

    return standardized_image

# Function to extract color features from an image
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Extract color features (e.g., mean values of each channel)
    mean_color = np.mean(image, axis=(0, 1))

    return mean_color

###############################################################################################################

# Path to image directories
# image_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/Parasitized'
# image_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ParasitizedTest'
# image_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/Uninfected'
image_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/UninfectedTest'

# Read a reference image
reference_image_path = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/Parasitized/p0.png'
reference_image = cv2.imread(reference_image_path)

# Initialize list to store standardized images
standardized_images = []
filenames = []

# Loop through the images in directory
for image_filename in os.listdir(image_directory):
    if image_filename.endswith('.png'):
        # Construct the full path to the image
        image_path = os.path.join(image_directory, image_filename)

        # Read the image
        image = cv2.imread(image_path)

        # Perform color standardization
        standardized_image = color_standardization(image, reference_image)

        # Append the standardized image to the list
        standardized_images.append(standardized_image)
        filenames.append(image_filename)

# Save standardized images
# output_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/ParasitizedTrain'
# output_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/ParasitizedTest'
# output_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/UninfectedTrain'
output_directory = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/UninfectedTest'

for i, standardized_image in enumerate(standardized_images):
    output_path = os.path.join(output_directory, filenames[i])
    cv2.imwrite(output_path, standardized_image)

# # Code to display the original and standardized images
# cv2.imshow('Original Image', image)
# cv2.imshow('Standardized Image', standardized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()