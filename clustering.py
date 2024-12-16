import os
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

st.header("Clustering Analysis:")

# Function to load random image from a directory
def load_random_image_from_directory(directory):
    if os.path.exists(directory):
        random_image_path = random.choice(os.listdir(directory))
        image = mpimg.imread(os.path.join(directory, random_image_path))
        return image
    else:
        st.warning(f"Directory not found: {directory}")
        return None

# Directories for the images
directories = {
    'Glioma Tumor': "../Brain tumor images/Testing/glioma_tumor",
    'Meningioma Tumor': "../Brain tumor images/Testing/meningioma_tumor",
    'No Tumor': "../Brain tumor images/Testing/no_tumor",
    'Pituitary Tumor': "../Brain tumor images/Testing/pituitary_tumor"
}

# Load images
images = {label: load_random_image_from_directory(directory) for label, directory in directories.items()}

# Filter out any None values (directories that weren't found)
images = {label: image for label, image in images.items() if image is not None}

# Check if any images were loaded
if not images:
    st.error("No images to process. Please check the directory paths.")
else:
    for label, image in images.items():
        # Flatten image to a 2D array of pixels and their RGB values
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Apply KMeans with fewer clusters
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(pixel_values)
        
        # Replace each pixel with the color of the centroid
        segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
        
        # Create a figure
        fig, axes = plt.subplots(1, 3, figsize=(15,7))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title(f'Original - {label}', fontsize=20)
        axes[0].axis('off')
        
        # Plot segmented image
        axes[1].imshow(segmented_img / 255.0)
        axes[1].set_title(f'Segmented - {label}', fontsize=20)
        axes[1].axis('off')
        
        # Plot KMeans cluster centers as scatter plot
        axes[2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c=kmeans.cluster_centers_ / 255.0)
        axes[2].set_title(f'KMeans Scatter - {label}', fontsize=20)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
