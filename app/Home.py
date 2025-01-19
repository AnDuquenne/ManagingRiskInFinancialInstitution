import streamlit as st
from PIL import Image

import sys, os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

st.header("Managing risk in financial institutions")
st.write("Be like...")
image = Image.open("app/granger.jpg")
st.image(image, caption="Granger Image")
