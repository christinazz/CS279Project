import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to extract color features from an image
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Extract color features (mean values of each channel)
    mean_color = np.mean(image, axis=(0, 1))

    return mean_color

# Function to extract color histograms from an image
def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space

    # Calculate 1D histogram for H channel (hue)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])

    # Normalize histogram
    hist_hue /= hist_hue.sum()

    return hist_hue.flatten()

###############################################################################################################

# CONSTRUCT TRAIN/TEST SETS

# Path to image directories
dir_p_train = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/ParasitizedTrain'
dir_p_test = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/ParasitizedTest'
dir_u_train = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/UninfectedTrain'
dir_u_test = '/Users/christinasze/Desktop/CS279/project/CS279Project/cell_images/images/ColorStandardizedImages/UninfectedTest'

# Initialize list to store filenames
filenames = []

# Construct train set
p_images_train = [os.path.join(dir_p_train, img) for img in os.listdir(dir_p_train) if img.endswith('.png')]
u_images_train = [os.path.join(dir_u_train, img) for img in os.listdir(dir_u_train) if img.endswith('.png')]
all_images_train = p_images_train + u_images_train

# Create train labels (1 for parasitic, 0 for uninfected)
y_train = np.concatenate([np.ones(len(p_images_train)), np.zeros(len(u_images_train))])

# Construct test set
p_images_test = [os.path.join(dir_p_test, img) for img in os.listdir(dir_p_test) if img.endswith('.png')]
u_images_test = [os.path.join(dir_u_test, img) for img in os.listdir(dir_u_test) if img.endswith('.png')]
all_images_test = p_images_test + u_images_test

# Create test labels (1 for parasitic, 0 for uninfected)
y_test = np.concatenate([np.ones(len(p_images_test)), np.zeros(len(u_images_test))])

###############################################################################################################

# # STRATEGY: MEAN COLOR VALUES

# print(f'Classifiers based on mean color value features')

# # train/test feature sets for mean color
# X_train = [extract_color_features(img) for img in all_images_train]
# X_test = [extract_color_features(img) for img in all_images_test]

# # Train SVM linear classifier
# linear_classifier = SVC(kernel='linear')
# linear_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = linear_classifier.predict(X_test)

# # Evaluate the model
# linear_accuracy = accuracy_score(y_test, y_pred)
# print(f'SVM Linear Classifier Accuracy: {linear_accuracy * 100:.2f}%')

# # Train SVM RBF classifier
# rbf_classifier = SVC(kernel='rbf')
# rbf_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rbf_classifier.predict(X_test)

# # Evaluate the model
# rbf_accuracy = accuracy_score(y_test, y_pred)
# print(f'SVM RBF Classifier Accuracy: {rbf_accuracy * 100:.2f}%')

# ###############################################################################################################

# # STRATEGY: COLOR HISTOGRAMS

# print(f'Classifiers based on color histogram features')

# # train/test feature sets for color histograms
# X_train = [extract_color_histogram(img) for img in all_images_train]
# X_test = [extract_color_histogram(img) for img in all_images_test]

# # Train SVM linear classifier
# linear_classifier = SVC(kernel='linear')
# linear_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = linear_classifier.predict(X_test)

# # Evaluate the model
# linear_accuracy = accuracy_score(y_test, y_pred)
# print(f'SVM Linear Classifier Accuracy: {linear_accuracy * 100:.2f}%')

# # Train SVM RBF classifier
# rbf_classifier = SVC(kernel='rbf')
# rbf_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rbf_classifier.predict(X_test)

# # Evaluate the model
# rbf_accuracy = accuracy_score(y_test, y_pred)
# print(f'SVM RBF Classifier Accuracy: {rbf_accuracy * 100:.2f}%')

###############################################################################################################

# STRATEGY: INCLUDE BOTH MEAN COLOR AND COLOR HISTOGRAM FEATURES

print(f'Classifiers based on mean color value feature and color histogram feature')

# train/test feature sets for mean color and color histograms
X_color_train = [extract_color_features(img) for img in all_images_train]
# X_color_test = [extract_color_features(img) for img in all_images_test]

print(np.shape(X_color_train))

X_hist_train = [extract_color_histogram(img) for img in all_images_train]

print(np.shape(X_hist_train))

print("Done")

X_hist_test = [extract_color_histogram(img) for img in all_images_test]

X_train = np.hstack((X_color_train, X_hist_train))
X_test = np.hstack((X_color_test, X_hist_test))

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))

# Train SVM linear classifier
linear_classifier = SVC(kernel='linear')
linear_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_classifier.predict(X_test)

# Evaluate the model
linear_accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Linear Classifier Accuracy: {linear_accuracy * 100:.2f}%')

# Train SVM RBF classifier
rbf_classifier = SVC(kernel='rbf')
rbf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rbf_classifier.predict(X_test)

# Evaluate the model
rbf_accuracy = accuracy_score(y_test, y_pred)
print(f'SVM RBF Classifier Accuracy: {rbf_accuracy * 100:.2f}%')