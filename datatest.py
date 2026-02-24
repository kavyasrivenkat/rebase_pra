import cv2
import os

# Folder paths
input_folder = "dataset"
output_folder = "processed_images/"
os.makedirs(output_folder, exist_ok=True)

# List of 9 images to process
images = ["Airplane.png","Baboon.png",  "Barbara.png", "Boat.png", "Couple.png", "Elaine.png", "Goldhill.png","Peppers.png","Zelda.png"]

for img_name in images:
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 512x512
    gray = cv2.resize(gray, (512, 512))
    # Save processed image
    cv2.imwrite(os.path.join(output_folder, img_name), gray)

print("Dataset prepared!")
