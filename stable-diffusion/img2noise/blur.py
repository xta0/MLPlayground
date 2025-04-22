import cv2

# Load the image
image = cv2.imread('./mona.jpg')

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (9, 9), 0)

# Save the blurred image
cv2.imwrite('blurred_image.jpg', blurred)

print("Blurred image saved as 'blurred_image.jpg'")