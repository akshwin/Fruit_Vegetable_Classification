import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model("FruitModel.keras")
labels = {0: 'Apple', 1: 'Banana', 2: 'Beetroot', 3: 'Bell pepper', 4: 'Cabbage', 5: 'Capsicum', 6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper', 9: 'Corn', 10: 'Cucumber', 11: 'Eggplant', 12: 'Garlic', 13: 'Ginger', 14: 'Grapes', 15: 'Jalepeno', 16: 'Kiwi', 17: 'Lemon', 18: 'Lettuce', 19: 'Mango', 20: 'Onion', 21: 'Orange',22: 'Paprika', 23: 'Pear', 24: 'Peas', 25: 'Pineapple',26: 'Pomegranate', 27: 'Potato', 28: 'Raddish', 29: 'Soy beans', 30: 'Spinach', 31: 'Sweetcorn', 32: 'Sweetpotato', 33: 'Tomato', 34: 'Turnip', 35: 'Watermelon'}
fruits = {'Banana', 'Apple', 'Pear', 'Grapes', 'Orange', 'Kiwi', 'Watermelon', 'Pomegranate', 'Pineapple', 'Mango'}
vegetables={'Cucumber', 'Carrot', 'Capsicum', 'Onion', 'Potato', 'Lemon', 'Tomato', 'Raddish', 'Beetroot', 'Cabbage', 'Lettuce', 'Spinach', 'Soy bean', 'Cauliflower', 'Bell pepper', 'Chilli pepper', 'Turnip', 'Corn', 'Sweetcorn', 'Sweet potato', 'Paprika', 'Jalepeño', 'Ginger', 'Garlic', 'Peas','Eggplant'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

def run():
    st.title("FRUITS🥭 AND VEGETABLE 🍅CLASSIFICATION ")
    img_file = st.file_uploader("Choose an image",type=['jpg','jpeg','png'])

    if img_file is not None :
        img  = Image.open(img_file).resize((250,250))
        st.image(img)
        save_image_path = './upload_image/'+img_file.name
        with open(save_image_path,"wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None :
            result = processed_img(save_image_path)
            if result in vegetables:
                st.info('**Category : Vegetable**')
            else :
                st.info('**Category : Fruit**')
            
            st.success("**Predicted : "+  result+"**")
run()
