# 🍎 Fruit & Vegetable Classifier App 

An interactive web app to classify **fruits and vegetables** from images using a **deep learning model** based on **VGG-16 + Transfer Learning**. The app is built using **Streamlit** and runs entirely in your browser – no setup required!

<p align="center">
  <a href="https://fruits-vegetables-classifier.streamlit.app/">
    <img src="https://img.shields.io/badge/🌐 Live%20App-Click%20Here-blue?style=for-the-badge" alt="Live App Badge"/>
  </a>
</p>

---

## 📌 Table of Contents

- [🧠 Abstract](#-abstract)
- [🚀 Key Features](#-key-features)
- [🍅 Classes Covered](#-classes-covered)
- [🧠 Methodology](#-methodology)
- [📈 Performance](#-performance)
- [⚙️ Installation & Usage](#️-installation--usage)
- [📁 Project Structure](#-project-structure)
- [🛣 Future Scope](#-future-scope)
- [📜 License & 📧 Contact](#-license--contact)

---

## 🧠 Abstract

The Fruit & Vegetable Classifier app enables users to upload or test sample images and classify them into **36 distinct categories** (fruits or vegetables). It uses a **fine-tuned VGG-16** convolutional model trained on an image dataset. With options for **detailed prediction display**, **model confidence**, and a clean UI, it’s suitable for educational demos, farming tech, and retail assistance.

---

## 🚀 Key Features

- 📤 Upload any `.jpg`, `.jpeg`, or `.png` image.
- 🍅 Classifies the image into one of 36 known food classes.
- 🧠 Differentiates between **fruit** and **vegetable** categories.
- 📈 Optionally shows **model confidence score**.
- 🧾 Sample image selector for instant testing.
- ⚙️ Powered by **VGG-16** and **Transfer Learning**.
- 🧼 Intuitive **Streamlit UI** with sidebar controls.

---

## 🍅 Classes Covered

### 🥦 Vegetables (25 Classes)
Beetroot, Bell Pepper, Cabbage, Capsicum, Carrot, Cauliflower, Chilli Pepper, Corn, Cucumber, Eggplant, Garlic, Ginger, Jalepeno, Lettuce, Onion, Paprika, Peas, Potato, Raddish, Soy Beans, Spinach, Sweetcorn, Sweetpotato, Tomato, Turnip

### 🍎 Fruits (11 Classes)
Apple, Banana, Grapes, Kiwi, Lemon, Mango, Orange, Pear, Pineapple, Pomegranate, Watermelon

---

## 🧠 Methodology

```plaintext
[ Uploaded Image or Sample ]
        │
        ▼
[ Resize to 224x224x3 ]
        │
        ▼
[ VGG-16 + Transfer Learning ]
        │
        ▼
[ Dense Layer + Softmax ]
        │
        ▼
[ Predicted Label + Category + Confidence Score ]
````

---

## 📈 Performance

* ✅ **Model**: VGG-16 (Keras Applications)
* 🧠 **Training Accuracy**: \~97%
* 🎯 **Validation Accuracy**: \~92%
* 🧪 **Top-1 Prediction**: Shown with confidence %

---

## ⚙️ Installation & Usage

### 🔧 Prerequisites

Make sure you have **Python 3.7+** installed.

### 📦 Install Dependencies

```bash
pip install streamlit pillow numpy tensorflow
```

Alternatively, use:

```bash
pip install -r requirements.txt
```

### 📥 Download the Model

Download `vgg.h5` and place it in the same directory as `app.py`.

### ▶️ Run the App

```bash
streamlit run app.py
```

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
fruit-veg-classifier/
├── app.py                # Streamlit app
├── vgg.h5                # Pre-trained VGG model
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── upload_image/         # Sample image folder
```

---

## 🛣 Future Scope

* 🔍 Add Grad-CAM visualizations for explainability
* 📱 Deploy as mobile app (using Kivy / React Native)
* 🎯 Add multi-label support for images with multiple items
* 🌐 Enable cloud-based inference for large-scale use

---

## 📜 License & 📧 Contact

**License**: MIT
**Author**: Akshwin T
📧 [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/) | [GitHub](https://github.com/akshwin)

---

<p align="center">
  ⭐ Found it useful? Star the repo and share it!
</p>

---