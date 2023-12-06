import cv2
import numpy as np

image_path = input("Provide path to image:")
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
threshold_value = 120
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
    cv2.drawContours(image, [cnt], 0, (0,255,0), 3)

if malaria_detected:
    print("Malaria Detected!")

    # Display the result
    cv2.imshow('Detected Malaria Dots', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No malaria present!")
