import cv2
import numpy as np
import os

def color_standardization(image, reference_image):
    # Convert images to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    # Calculate mean and standard deviation for each channel in LAB color space
    mean_image, stddev_image = cv2.meanStdDev(image_lab)
    mean_reference, stddev_reference = cv2.meanStdDev(reference_lab)

    # Perform color standardization
    standardized_image = image_lab.copy()
    for i in range(3):  # Iterate over LAB channels
        standardized_image[:, :, i] = ((image_lab[:, :, i] - mean_image[i]) * (stddev_reference[i] / stddev_image[i])) + mean_reference[i]

    # Convert back to BGR color space
    standardized_image = cv2.cvtColor(standardized_image, cv2.COLOR_LAB2BGR)

    return standardized_image

# Read an image and a reference image
image_path = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/Parasitized/p5.png'
reference_image_path = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/Parasitized/p0.png'

image = cv2.imread(image_path)
reference_image = cv2.imread(reference_image_path)

# Perform color standardization
standardized_image = color_standardization(image, reference_image)

# Display the original and standardized images
cv2.imshow('Original Image', image)
cv2.imshow('Standardized Image', standardized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()