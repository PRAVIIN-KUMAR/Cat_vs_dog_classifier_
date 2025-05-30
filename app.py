import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model # Used to load the .h5 model
from PIL import Image
import numpy as np

# --- Page Configuration (optional, but good for custom titles/icons) ---
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered" # Can be "wide" for more space
)

# --- Load the trained model ---
# Ensure 'cats_vs_dogs_model.h5' is in the same directory as this app.py file
MODEL_PATH = 'cats_vs_dogs_model.h5'
try:
    # Use st.cache_resource to load the model only once
    @st.cache_resource
    def load_my_model():
        model = load_model(MODEL_PATH)
        return model

    model = load_my_model()
    st.success("Model loaded successfully! Ready for predictions.")
except Exception as e:
    st.error(f"🚨 Error loading the model: {e}")
    st.info("Please ensure 'cats_vs_dogs_model.h5' is in the same directory and is a valid Keras model file.")
    st.stop() # Stop execution if model loading fails

# --- Main Page Content ---
st.title("🐱🐶 Cat vs Dog Image Classifier")

st.markdown("""
Welcome to the Cat vs Dog Image Classifier! This application uses a deep learning model to predict whether an uploaded image contains a cat or a dog.
Simply upload an image below and let the AI do the magic!
""")

uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.markdown("### Analyzing your image... 🤖")

    # Preprocess the image to match the model's expected input
    img_array = np.array(image)
    # Convert image to RGB if it's grayscale or RGBA
    if img_array.shape[-1] == 4: # RGBA
        img_array = img_array[..., :3]
    elif len(img_array.shape) == 2: # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)

    img = tf.image.resize(img_array, (256, 256)) # Resize to 256x256 as used in the notebook
    img = tf.cast(img / 255.0, tf.float32) # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0) # Add a batch dimension (model expects shape like (1, 256, 256, 3))

    # Make prediction
    prediction = model.predict(img)

    # Interpret and display the prediction
    # Assuming output is a single value between 0 and 1
    st.write("---") # Separator
    if prediction[0][0] > 0.5:
        st.markdown(f"## Prediction: <span style='color:green;'>🐶 It's a Dog!</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"## Prediction: <span style='color:blue;'>🐱 It's a Cat!</span>", unsafe_allow_html=True)
    st.write(f"Confidence: `{prediction[0][0]:.2f}`")
    st.write("---") # Separator

    st.info("The model outputs a value between 0 and 1. Values closer to 0 indicate 'Cat', and values closer to 1 indicate 'Dog'.")


# --- Sidebar Content ---
st.sidebar.header("🐾 About the Model")
st.sidebar.markdown("""
This application leverages a **Convolutional Neural Network (CNN)**, a powerful type of deep learning model, specifically designed for image classification tasks.
""")

st.sidebar.subheader("💡 How it Works:")
st.sidebar.markdown("""
1.  **Image Upload:** You upload an image (JPG, JPEG, PNG).
2.  **Preprocessing:** The image is resized to 256x256 pixels and its pixel values are normalized to be between 0 and 1.
3.  **Prediction:** The preprocessed image is fed into our trained CNN model.
4.  **Result:** The model outputs a probability score, which is then translated into a "Cat" or "Dog" prediction with a confidence level.
""")

st.sidebar.subheader("🧠 Model Details:")
st.sidebar.markdown("""
* **Framework:** TensorFlow 2.x with Keras API
* **Architecture:** Custom CNN with multiple convolutional layers, pooling layers, and dense layers.
* **Dataset:** Trained on a large subset of the **Dogs vs. Cats dataset** from Kaggle.
* **Input Size:** 256x256 pixels with 3 color channels (RGB).
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using Streamlit")
