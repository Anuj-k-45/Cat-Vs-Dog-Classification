import streamlit as st
from PIL import Image
import numpy as np
import pickle

model = pickle.load(open("CatVsDog.pkl", "rb"))

st.title("CATS VS DOGS CLASSIFIER")

st.write("Upload an Image")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:

    image = Image.open(uploaded_image).resize((256, 256))
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    image = image.convert("RGB")
    img_arr = np.array(image)
    img_arr = img_arr/255

    test_input = img_arr.reshape(1, 256, 256, 3) 

    y_pred = model.predict(test_input)
    st.write(y_pred)
    if(y_pred[0][0]<0.5):
        st.write("Prediction : It is a CAT")
    else:
        st.write("Prediction : It is a DOG")