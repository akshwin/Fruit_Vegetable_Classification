import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import os

# Load the trained model
model = load_model("mobilenet_model.h5", compile=False)

# Label mappings
labels = {
    0: 'Apple', 1: 'Banana', 2: 'Beetroot', 3: 'Bell pepper', 4: 'Cabbage', 5: 'Capsicum',
    6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper', 9: 'Corn', 10: 'Cucumber', 11: 'Eggplant',
    12: 'Garlic', 13: 'Ginger', 14: 'Grapes', 15: 'Jalepeno', 16: 'Kiwi', 17: 'Lemon',
    18: 'Lettuce', 19: 'Mango', 20: 'Onion', 21: 'Orange', 22: 'Paprika', 23: 'Pear',
    24: 'Peas', 25: 'Pineapple', 26: 'Pomegranate', 27: 'Potato', 28: 'Raddish',
    29: 'Soy beans', 30: 'Spinach', 31: 'Sweetcorn', 32: 'Sweetpotato', 33: 'Tomato',
    34: 'Turnip', 35: 'Watermelon'
}

fruits = {
    'Banana', 'Apple', 'Pear', 'Grapes', 'Orange', 'Kiwi', 'Watermelon',
    'Pomegranate', 'Pineapple', 'Mango'
}

vegetables = {
    'Cucumber', 'Carrot', 'Capsicum', 'Onion', 'Potato', 'Lemon', 'Tomato', 'Raddish',
    'Beetroot', 'Cabbage', 'Lettuce', 'Spinach', 'Soy beans', 'Cauliflower', 'Bell pepper',
    'Chilli pepper', 'Turnip', 'Corn', 'Sweetcorn', 'Sweetpotato', 'Paprika',
    'Jalepeno', 'Ginger', 'Garlic', 'Peas', 'Eggplant'
}

# Classify uploaded image
def classify_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class.capitalize()

# Streamlit app
def run():
    st.set_page_config(page_title="Fruit & Vegetable Classifier", layout="centered")

    # Sidebar
    st.sidebar.title("Fruit & Veggie Classifier 🥦")
    st.sidebar.markdown("""
    - This project uses a deep learning model to classify images of fruits and vegetables into 36 categories.
    
    - The model is developed with the help of the MobileNetV2 Architecture and trained on a dataset of 36 classes.
    
    **Steps to use:**
    - Upload an image
    - View prediction and category
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 Developed by Akshwin T ")
    st.sidebar.markdown("📬 Contact: [akshwint.2003@gmail.com](mailto:youremail@example.com)")

    # Main area
    st.title("🍎 Fruit & Vegetable Classifier🥦")
    st.write("Upload a clear image of a fruit or vegetable, and the model will tell you what it is!")
    
    img_file = st.file_uploader("📤 Upload an MRI Image", type=['jpg', 'jpeg', 'png'])

    if img_file is not None:
        upload_dir = "./upload_image"
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, img_file.name)

        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Show uploaded image
        # Centered and resized image display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(Image.open(save_path), caption='🖼 Uploaded MRI Scan', width=300)

        # Prediction
        prediction = classify_image(save_path)
        category = "Vegetable" if prediction in vegetables else "Fruit"

        # Results
        st.markdown("### 🔍 Prediction Result")
        st.info(f"**Category**: {category}")
        st.success(f"**Predicted**: {prediction}")

# Run the app
if __name__ == "__main__":
    run()
    