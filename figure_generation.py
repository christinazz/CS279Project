import cv2
import numpy as np
import os
import shutil
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

labels = ["Uninfected", "Infected"]

######################################################################################################################

# CV Thresholding Classifier CM

cv_cm = [[2573, 178], 
         [354, 2389]]

# Plotting using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cv_cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=labels)

# Adding labels and title for clarity
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Display the plot
plt.show()

######################################################################################################################

celltool_only_cm = [[1977, 774],
                    [1364, 1379]]

# Plotting using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(celltool_only_cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=labels)

# Adding labels and title for clarity
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Display the plot
plt.show()

######################################################################################################################

final_svm_cm = [[2585, 166],
                [200, 2543]]

# Plotting using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(final_svm_cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=labels)

# Adding labels and title for clarity
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Display the plot
plt.show()

######################################################################################################################

final_lr_cm = [[2575, 176],
               [205, 2538]]

# Plotting using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(final_lr_cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=labels)

# Adding labels and title for clarity
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Display the plot
plt.show()

######################################################################################################################

# Color confusion matrices
color_cm = [[2530, 221],
            [ 439, 2304]]

# Plotting using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(color_cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=labels)

# Adding labels and title for clarity
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Display the plot
plt.show()