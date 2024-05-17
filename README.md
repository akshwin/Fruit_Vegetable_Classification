# Fruit and Vegetable Classification App 🍏🍅

## Overview

This Streamlit app uses a pre-trained deep learning model to classify whether an uploaded image contains a fruit or a vegetable. The model has been trained on a variety of fruits and vegetables.

## Getting Started

To run the app locally, follow these steps:

1. Install the required libraries:

    ```bash
    pip install streamlit pillow numpy tensorflow
    ```

2. Download the pre-trained model file (`FruitModel.h5`) and save it in the project directory.

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

4. The app will open in your default web browser. Upload an image of a fruit or vegetable to see the classification result.

## Model Information

The deep learning model (`FruitModel.h5`) used in this app is trained to recognize the following classes:

- Apple
- Banana
- Beetroot
- Bell pepper
- Cabbage
- Capsicum
- Carrot
- Cauliflower
- Chilli pepper
- Corn
- Cucumber
- Eggplant
- Garlic
- Ginger
- Grapes
- Jalepeno
- Kiwi
- Lemon
- Lettuce
- Mango
- Onion
- Orange
- Paprika
- Pear
- Peas
- Pineapple
- Pomegranate
- Potato
- Raddish
- Soy beans
- Spinach
- Sweetcorn
- Sweetpotato
- Tomato
- Turnip
- Watermelon

## Usage

1. Open the app in your web browser.
2. Upload an image using the "Choose an image" button.
3. The app will display the uploaded image and provide the predicted category (Fruit or Vegetable).
4. The specific predicted label will also be shown (e.g., "Predicted: Apple" or "Predicted: Tomato").

## To run the Application Online 

https://fruit-vegetable-detector.streamlit.app/
