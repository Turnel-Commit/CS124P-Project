import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("Dataset/leaf.jpeg")
image = cv2.resize(image, (600, 800))

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# -----------------------------
# Detect Leaf Area (Green)
# -----------------------------
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

# Clean mask
kernel = np.ones((5,5), np.uint8)
leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# Detect Disease Area (Brown)
# -----------------------------
lower_brown = np.array([10, 50, 50])
upper_brown = np.array([30, 255, 255])

disease_mask = cv2.inRange(hsv, lower_brown, upper_brown)
disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# Contour Detection
# -----------------------------
contours, _ = cv2.findContours(disease_mask, 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)

total_disease_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:  # remove small noise
        total_disease_area += area
        cv2.drawContours(image, [cnt], -1, (0,140,255), 2)

# -----------------------------
# Measure Areas
# -----------------------------
leaf_area = cv2.countNonZero(leaf_mask)
infection_percent = (total_disease_area / leaf_area) * 100

# Display results
print("Leaf Area:", leaf_area)
print("Disease Area:", total_disease_area)
print("Infection %:", infection_percent)

cv2.putText(image, f"Infection: {infection_percent:.2f}%",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,0,0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Disease Detection")
plt.axis("off")
plt.show()