import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load the pre-trained model
model_path = 'my_model.h5'
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit app
st.title("Image Classification App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(img_array)

    # Convert NumPy array to a tuple for using as a key
    tuple_key = tuple(prediction.flatten())

    # Dictionary mapping index to class names
    class_names = {0: 'cane', 1: 'cavallo', 2: 'elefante', 3: 'farfalla', 4: 'gallina', 5: 'gatto', 6: 'mucca', 7: 'pecora', 8: 'ragno', 9: 'scoiattolo'}

    # Display the result
    st.write(f"Predicted Class: {class_names[np.argmax(prediction)]}")