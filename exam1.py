import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('your_image_path.jpg')

# Check if the image is loaded properly
if image is None:
    print("Could not open or find the image")
    exit(0)

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.show()

# Separate the channels
b, g, r = cv2.split(image_rgb)

# Display each channel in a different window
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1), plt.imshow(b, cmap='gray'), plt.title("Blue Channel")
plt.subplot(1, 3, 2), plt.imshow(g, cmap='gray'), plt.title("Green Channel")
plt.subplot(1, 3, 3), plt.imshow(r, cmap='gray'), plt.title("Red Channel")
plt.show()

# Convert the image to grayscale (black and white)
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Display the grayscale image
plt.figure(figsize=(10, 10))
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.show()

