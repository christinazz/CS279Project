import cv2
import numpy as np
import os
import shutil
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Directory paths
parasitized_dir = 'cell_images/images/ParasitizedTest'
uninfected_dir = 'cell_images/images/UninfectedTest'
combined_dir = 'cell_images/images/CVTest'

def copy_and_rename(src_dir, dst_dir, prefix):
	for filename in os.listdir(src_dir):
		new_filename = f"{prefix}_{filename}"
		shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, new_filename))

# Copy and rename test files to combined directory
if not os.path.exists(combined_dir):
	os.makedirs(combined_dir)
	copy_and_rename(parasitized_dir, combined_dir, 'Parasitized')
	copy_and_rename(uninfected_dir, combined_dir, 'Uninfected')

total_images = 0
correct_predictions = 0
y_true = [] # Actual labels
y_pred = [] # Predicted labels

for filename in os.listdir(combined_dir):
	if filename.lower().endswith('.png'):
		total_images += 1
		image_path = os.path.join(combined_dir, filename)
		image = cv2.imread(image_path)

		# Convert to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Apply threshold
		# threshold_value = 120
		threshold_value = 128
		ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

		# Morphological operations
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

		# Detect contours
		contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Image dimensions
		height, width = image.shape[:2]

		# Loop through contours and ignore those touching the image border
		malaria_detected = False
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			if x <= 1 or y <= 1 or (x + w) >= width-1 or (y + h) >= height-1:
				continue  # Ignore contours touching the image border
			
			# Optional: Check the area of the contour if needed
    	# area = cv2.contourArea(cnt)
    	# if area_min < area < area_max:
			malaria_detected = True
			# cv2.drawContours(image, [cnt], 0, (0,255,0), 3)

		actual_label = 'Parasitized' if 'Parasitized' in filename else 'Uninfected'
		y_true.append(actual_label)

		predicted_label = 'Parasitized' if malaria_detected else 'Uninfected'
		y_pred.append(predicted_label)

		if actual_label == predicted_label:
			correct_predictions += 1
print("Total images:", total_images)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=["Parasitized", "Uninfected"])
print("Confusion Matrix:")
print(conf_matrix)

# Calculate Accuracy
accuracy = correct_predictions / total_images * 100
print(f"Accuracy: {accuracy:.2f}%")

# Calculate Sensitivity and Specificity
tp = conf_matrix[0][0]
fn = conf_matrix[0][1]
tn = conf_matrix[1][1]
fp = conf_matrix[1][0]

sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print(f"Sensitivity: {sensitivity:.2f}%")
print(f"Specificity: {specificity:.2f}%")

# Precision, Recall, and F1 Score calculations
precision = tp / (tp + fp) * 100
recall = sensitivity
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1_score:.2f}%")

labels = list(set(y_true))  # Extract the labels from your true labels list

# Plotting using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=labels)

# Adding labels and title for clarity
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Display the plot
plt.show()


## Threshold value: 120
# Confusion Matrix:
# [[2263  480]
#  [  83 2668]]
# Accuracy: 89.75%
# Sensitivity: 82.50%
# Specificity: 96.98%

## Threshold value: 100
# Confusion Matrix:
# [[1579 1164]
#  [   9 2742]]
# Accuracy: 78.65%
# Sensitivity: 57.56%
# Specificity: 99.67%

## Threshold value: 110
# Confusion Matrix:
# [[1988  755]
#  [  33 2718]]
# Accuracy: 85.66%
# Sensitivity: 72.48%
# Specificity: 98.80%

## Threshold value: 140
# Confusion Matrix:
# [[2411  332]
#  [ 362 2389]]
# Accuracy: 87.37%
# Sensitivity: 87.90%
# Specificity: 86.84%

## Threshold value: 130
# Confusion Matrix:
# [[2401  342]
#  [ 198 2553]]
# Accuracy: 90.17%
# Sensitivity: 87.53%
# Specificity: 92.80%

## Threshold value: 135
# Confusion Matrix:
# [[2426  317]
#  [ 276 2475]]
# Accuracy: 89.21%
# Sensitivity: 88.44%
# Specificity: 89.97%

## Threshold value: 133
# Confusion Matrix:
# [[2426  317]
#  [ 250 2501]]
# Accuracy: 89.68%
# Sensitivity: 88.44%
# Specificity: 90.91%

## Threshold value: 128
# Confusion Matrix:
# [[2389  354]
#  [ 178 2573]]
# Accuracy: 90.32%
# Sensitivity: 87.09%
# Specificity: 93.53%
