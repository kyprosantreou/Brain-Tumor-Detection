import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt  

st.header("Upload your MRI scan for tumor detection and prediction")

# Load model
model = models.load_model("effnet.keras")

# Define class names
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor", "other"]

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the image file as bytes and convert it to a format compatible with cv2
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    try:
        if img is None:
            raise FileNotFoundError("Error: Could not load the image. Check the file path or file format.")
        
        # Preprocess the image
        img = cv2.resize(img, (100, 100))
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img_array = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img_array)
        percentage_prediction = prediction * 100

        # Display results
        index = np.argmax(prediction)

        class_names = ["Glioma tumor", "Meningioma tumor", "No tumor", "Pituitary tumor","Its not a brain MRI image"]
            
        st.subheader(f"Prediction is: {class_names[index]}")

        for i, pred in enumerate(percentage_prediction[0]):
            st.write(f"{class_names[i]}: {pred:.2f}%")

        # Create Grad-CAM model
        last_conv_layer_name = 'top_conv'  # Replace with correct layer name if needed
        heatmap_model = tf.keras.models.Model(
            inputs=model.input, 
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = heatmap_model(img_array)
            loss = predictions[:, np.argmax(predictions)]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        # Generate the heatmap
        heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        # Apply heatmap to the image
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        # Create figure for displaying the images
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Overlayed image (now in position 2)
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title("Overlayed Image")
        plt.axis("off")

        # Heatmap image (now in position 3)
        plt.subplot(1, 3, 3)
        heatmap_normalized = heatmap / 255.0  # Normalize values to [0, 1] for better understanding
        heatmap_plot = plt.imshow(heatmap_normalized, cmap='jet')  # Plot the heatmap for the color bar
        plt.title("Heatmap")
        plt.axis("off")

        # Add color bar
        cbar = plt.colorbar(heatmap_plot)  # Associate the color bar with the heatmap
        cbar.set_label("Activation Intensity", rotation=270, labelpad=15)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['Low', 'Medium', 'High'])

        plt.tight_layout()
        
        # Display the figure in Streamlit
        st.pyplot(plt)

    except FileNotFoundError as e:
        st.error(str(e))
