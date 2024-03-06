#Build a UI for the fastapi app using streamlit

import os
import streamlit as st
import requests

st.title("Text Classification App")
st.write("This is a simple text classification app")

text = st.text_area("Enter the text to classify", "Type Here")

if st.button("Classify"):
    result = requests.get(f"https://16.16.28.221:8080/predict?text={text}")
    st.write(result.text)

# if st.button("Train Model"):
#     result = requests.get(f"http://13.49.72.14:8080/train")