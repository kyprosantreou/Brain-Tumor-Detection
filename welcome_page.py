import streamlit as st

st.markdown("""
    # About the Brain Tumor Detection App:

    Welcome to the **Brain Tumor Detection App**, a powerful tool designed to assist in the early detection of brain 
    tumors using advanced deep learning techniques. This app leverages a trained convolutional neural network (CNN) model to 
    analyze MRI images and classify them into one of four categories:

    - Glioma Tumor
    - Meningioma Tumor
    - No Tumor
    - Pituitary Tumor

    The app provides an easy-to-use interface where users can upload MRI images, and the model will return a prediction of the tumor type, 
    if present, along with a confidence percentage. Additionally, the app visualizes the model's attention to certain regions of the image using a Grad-CAM heatmap, which highlights the areas that contributed most to the prediction.

    ### How to Use:
    #### Brain MRI Anlaysis page
    1. **Upload an Image**: Click on the "Choose an image" button and select an MRI image from your device. 
        The image should be in JPG, JPEG, or PNG format.
       
    2. **Get Prediction**: After uploading the image, the app will process it and provide a prediction. 
        It will show the type of tumor (if detected) and a percentage confidence for each class.

    3. **Visualize Results**: The app will also generate a heatmap overlay, which highlights the areas of the MRI scan that influenced the model’s 
        decision. The heatmap helps visualize the regions of interest, giving insights into the model's reasoning.

    #### Clustering page
    This app performs clustering analysis on brain tumor images using KMeans. It allows users to visualize original and segmented versions of images 
    from four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor. By leveraging KMeans clustering, the app highlights distinct 
    color regions in the images, offering a simplified perspective of complex medical scans.    
              
    ### What’s Behind the App?

    The model powering this app is based on **EfficientNet**, a state-of-the-art deep learning model for image classification, 
    which has been trained on a dataset of brain tumor MRI images. The app uses TensorFlow and Keras to load the trained model and 
    perform predictions.
            
    """)