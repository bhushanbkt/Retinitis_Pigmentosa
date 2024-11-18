import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load your trained model (ensure the model is in the same directory or provide the correct path)
model = tf.keras.models.load_model('Retinitis_Pigmentosa.h5')

# Define image size for the model
IMG_SIZE = (224, 224)

# Function to preprocess a single image for prediction
def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize(IMG_SIZE)
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to generate a synthetic ground truth mask using thresholding
def generate_ground_truth_mask(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply thresholding to simulate a ground truth mask (adjust threshold as needed)
    _, ground_truth_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Convert the mask to RGB for visualization (as a 3-channel mask)
    ground_truth_mask = cv2.cvtColor(ground_truth_image, cv2.COLOR_GRAY2RGB)
    return ground_truth_mask

# Function to display the predicted segmented image
def display_prediction(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    
    # Generate prediction (predicted mask) using the model
    predicted_mask = model.predict(processed_image)
    predicted_mask = np.squeeze(predicted_mask)  # Remove extra dimensions

    # Generate ground truth mask using synthetic thresholding
    ground_truth_mask = generate_ground_truth_mask(image)

    # Plot the original image, ground truth mask, and predicted mask
    st.image(image, caption="Original Image", use_column_width=True)

    # Apply color map to the predicted mask for better visualization
    cmap_predicted_mask = plt.get_cmap('viridis')(predicted_mask)[..., :3]  # Apply color map, discard alpha

    # Display the ground truth mask
    st.image(ground_truth_mask, caption="Simulated Ground Truth Mask", use_column_width=True)

    # Display the predicted mask as a segmented image
    st.image(cmap_predicted_mask, caption="Predicted Segmented Mask", use_column_width=True)

# Streamlit UI
st.title("Image Segmentation with Deep Learning")

# Image upload functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Display the predicted segmented image when the user clicks the button
    if st.button("Predict Segmentation"):
        display_prediction(uploaded_file)
