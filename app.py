import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = 256
CHANNELS = 3
class_names = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

# Load the model
model = tf.keras.models.load_model("Model/Model/1.keras")

# Information on each disease
disease_info = {
    "Potato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani, which thrives in warm, humid conditions.",
        "prevention": "Use disease-free seeds, rotate crops, apply fungicides, and avoid overhead irrigation."
    },
    "Potato___Late_blight": {
        "cause": "Caused by the water mold Phytophthora infestans, which spreads in cool, wet conditions.",
        "prevention": "Use resistant potato varieties, ensure good air circulation, and remove infected plants promptly."
    }
}

# Prediction function
def predict_image(image, model, class_names):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Custom CSS for background and text styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        h1 {
            color: #2e7d32;
            text-align: center;
            font-family: 'Arial Black', sans-serif;
        }
        .intro-text {
            text-align: center;
            font-size: 1.1em;
            color: #444444;
        }
        .info-box {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .info-box h3 {
            color: #2e7d32;
        }
        .prediction-result {
            color: #1b5e20;
            font-size: 1.3em;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App UI
st.title("Potato Disease Classification 🌿")
st.markdown("<p class='intro-text'>Upload an image of a potato leaf to classify its health or detect diseases.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify button
    if st.button("🔍 Classify Image"):
        predicted_class, confidence = predict_image(image, model, class_names)
        
        # Display prediction result
        st.markdown(f"<p class='prediction-result'>Predicted Class: <strong>{predicted_class}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p class='prediction-result'>Confidence: <strong>{confidence}%</strong></p>", unsafe_allow_html=True)
        
        # Display additional disease information if the prediction is a disease
        if predicted_class != "Potato___healthy":
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown(f"<h3>Disease Information</h3>", unsafe_allow_html=True)
            st.markdown(f"**Cause**: {disease_info[predicted_class]['cause']}")
            st.markdown(f"**Prevention**: {disease_info[predicted_class]['prevention']}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success("The plant appears healthy! No disease detected.")
