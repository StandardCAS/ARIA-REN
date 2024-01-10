import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_3d(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Convert the grayscale image into a numpy array
    img_array = np.array(img_gray)

    # Create x and y coordinates
    x = np.linspace(0, img_array.shape[1], img_array.shape[1])
    y = np.linspace(0, img_array.shape[0], img_array.shape[0])
    x, y = np.meshgrid(x, y)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, img_array, cmap='gray')

    # Set the aspect ratio of the axes
    aspect_ratio = img_array.shape[1] / img_array.shape[0]
    ax.auto_scale_xyz([0, img_array.shape[1]], [0, img_array.shape[0]], [0, img_array.max()])
    ax.set_box_aspect([aspect_ratio, 1, 0.5])  # Adjust the last value to change the vertical scale
    # Show the plot
    plt.show()

# Use the function
plot_3d('Screenshot 2024-01-10 at 10.00.33 PM.png')
