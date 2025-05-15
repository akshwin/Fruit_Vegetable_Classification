# 🍏 Fruit & Vegetable Classifier App

A **Streamlit-powered web application** that uses a **deep learning model** to classify images of fruits and vegetables. Upload any image and let the AI instantly identify the item and tell you whether it's a **fruit** or a **vegetable**, along with the specific name (e.g., Apple, Carrot, Lettuce).

---

## 🚀 Quick Start

### 🔧 Installation

Ensure you have Python installed. Then run the following:

```bash
pip install streamlit pillow numpy tensorflow
```

### 📁 Download the Model

* Download the pre-trained model file: **`model.h5`**
* Place it in the root directory of your project (same location as `app.py`)

### ▶️ Launch the App

```bash
streamlit run app.py
```

The app will launch in your browser at: `http://localhost:8501`

---

## 🧠 Model Overview

This classifier is built using a **Convolutional Neural Network (CNN)** trained on a labeled dataset of various fruits and vegetables. It has been trained to recognize the following **36 classes**:

### 🥦 Vegetables

* Beetroot
* Bell Pepper
* Cabbage
* Capsicum
* Carrot
* Cauliflower
* Chilli Pepper
* Corn
* Cucumber
* Eggplant
* Garlic
* Ginger
* Jalepeno
* Lettuce
* Onion
* Paprika
* Peas
* Potato
* Raddish
* Soy Beans
* Spinach
* Sweetcorn
* Sweetpotato
* Tomato
* Turnip

### 🍎 Fruits

* Apple
* Banana
* Grapes
* Kiwi
* Lemon
* Mango
* Orange
* Pear
* Pineapple
* Pomegranate
* Watermelon

---

## 🖼️ How to Use

1. Launch the app and open it in your browser.
2. Upload an image (`.jpg`, `.jpeg`, or `.png`) using the upload box.
3. The app will:

   * Display the uploaded image
   * Predict whether the object is a fruit or vegetable
   * Show the **exact class name** (e.g., "🍋 Lemon", "🥕 Carrot")
   * Provide the model’s confidence score

---

## 🌐 Try It Online

No setup needed — launch the app directly in your browser:

👉 [**Live App**](https://fruit-vegetable-detector.streamlit.app/)

---

## 📌 Project Structure

```plaintext
├── app.py                # Streamlit app file
├── FruitModel.h5         # Pre-trained Keras model
├── requirements.txt      # Optional: to specify all dependencies
└── README.md             # This file
```

---

## 📬 Contact

👨‍💻 Developed by **Akshwin T**
📧 Email: [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)

---