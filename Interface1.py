# ---- Import Necessary Python Modules ----
import streamlit as st
import fitz
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.h5") # load
LABELS = ['A Medical Image', 'Not A Medical Image']

# preprocess image ---
def preprocess_image(img):
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_class(img):
    arr = preprocess_image(img)
    pred = model.predict(arr)[0][0]
    label = LABELS[int(pred > 0.5)]
    return label, pred

# --- Extract images from PDF ---
def extract_images_from_pdf(uploaded_file):
    images = []
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        for img in page.get_images(full=True):
            base_img = doc.extract_image(img[0])
            img_bytes = base_img["image"]
            img = Image.open(BytesIO(img_bytes))
            images.append(img)
    return images

# --- Extract images from URL ---
def extract_images_from_url(url):
    images = []
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src")
            if src and src.startswith("http"):
                img_resp = requests.get(src)
                img = Image.open(BytesIO(img_resp.content))
                images.append(img)
    except Exception as e:
        st.error(f"Failed to fetch images: {e}")
    return images

# --- Streamlit User Interface ---
st.title("🧠 CNN Image Classifier from URL")
# option = st.radio("Choose input type:", ["Upload PDF", "Enter Web URL"])
option = st.radio("Choose input type:", ["Enter Web URL"])

if option == "Enter Web URL":
    url = st.text_input("Paste URL that contains images")
    if url:
        st.info("Fetching images from URL...")
        images = extract_images_from_url(url)
        if not images:
            st.warning("No valid images found on this webpage.")
        for img in images:
            st.image(img, caption="Extracted from URL", use_column_width=True)
            label, prob = predict_class(img)
            st.success(f"Prediction: **{label}** ({prob:.2f})")
else:

    uploaded_file = st.file_uploader("Upload PDF with images", type="pdf")
    if uploaded_file:
        st.info("Extracting images from PDF...")
        images = extract_images_from_pdf(uploaded_file)
        if not images:
            st.warning("No images found in PDF.")
        for img in images:
            st.image(img, caption="Extracted Image", use_column_width=True)
            label, prob = predict_class(img)
            st.success(f"Prediction: **{label}** ({prob:.2f})")
