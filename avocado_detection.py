import glob
import streamlit as st
import wget
from PIL import Image
import torch
import os
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Avocado Defect Detection",
    page_icon=":avocado:",
    layout="wide"
)

cfg_model_path = 'models/resnet50.pt'
model = None
confidence = .10

def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.",
                               min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img, results = infer_image(img_file)
            st.image(img, caption="Model prediction")
            
            # Extracting the detected defects and their confidence scores
            detected_defects = results.pandas().xyxy[0][['name', 'confidence']]

        df = pd.DataFrame(detected_defects)
        df.columns = ['Avocados Type', 'Confidence']  # Renaming the columns for clarity
        st.table(df)

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    
    # Convert image to writable format
    for i in range(len(result.ims)):
        result.ims[i] = result.ims[i].copy()

    result.render()  # This modifies the image directly with annotations
    image = Image.fromarray(result.ims[0])
    return image, result

@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('WangRongsheng/BestYOLO',
                            'custom', path=path, force_reload=True)
    model_.to(device)
    return model_

@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file

def main():
    global model, confidence, cfg_model_path

    st.title("Avocado Defect Detection :avocado:")
    st.sidebar.image("assets/undipa.png", width=200)
    st.sidebar.write("Welcome to Avocado Defect Detection :avocado:")
    st.sidebar.divider()

    st.sidebar.title(":gear: Settings")
    st.sidebar.write("Please Configure Settings")

    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available! Please add it to the model folder.", icon=":warning:")
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], index=0)
        else:
            device_option = "cpu"

        model = load_model(cfg_model_path, device_option)

        input_option = st.sidebar.selectbox(
            "Select Activity: ", ['🎪 Home', 
                                  '📷 Detect Defects in Image'], 
                                  index=0)
        st.sidebar.markdown("---")

        if input_option == '📷 Detect Defects in Image':
            confidence = st.sidebar.slider(':eye-in-speech-bubble: Confidence Level', min_value=0.1, max_value=0.8, value=.10)

            if st.sidebar.checkbox(":hammer_and_pick: Custom Classes"):
                model_names = list(model.names.values())
                assigned_class = st.sidebar.multiselect(
                    "Select Classes", model_names, default=[model_names[0]])
                classes = [model_names.index(name) for name in assigned_class]
                model.classes = classes
            else:
                model.classes = list(model.names.keys())
            st.sidebar.markdown("---")

            data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
            image_input(data_src)

        elif input_option == '🎪 Home':
            html_temp_home1 = """<div style="background-color:#12d33b;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Avocado Quality Defects Detection application</h4>
                                            </div>
                                            </br>"""
            st.markdown(html_temp_home1, unsafe_allow_html=True)
            st.write("""
                    Features of this App: 
                    1. Home
                    2. Feature to detect avocado defects by Image
                    """)
            
            html_temp_home2 = """<hr style="height:10px;border:none;background-color:#12d33b;" />"""
            st.markdown(html_temp_home2, unsafe_allow_html=True)

            html_temp_home3 = """<h1 style='text-align: center; color: black;'>Common Avocado Defects</h1></br>"""
            st.markdown(html_temp_home3, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image("assets/suffocation.jpg", caption="Suffocation Avocado Image")
            with col2:
                st.subheader("Suffocation")
                st.write("Carbon-dioxide poisoning or Suffocation occurs in avocados due to a malfunction of the controlled atmosphere system which causes the fruit to die and limp and generally occurs when the fruit is ripe, also develops during transit. Symptoms experienced by this defective fruit have dark brown to black circular lesions and appear shiny throughout the fruit. Another cause of this defect occurs due to variable fruit packaging, where older and more ripe fruit is mixed with younger and less ripe fruit.")

            col3, col4 = st.columns([1, 2])
            with col3:
                st.image("assets/damaged.jpeg", caption="Physical Damage Avocado Image")
            with col4:
                st.subheader("Physical Damage")
                st.write("This type of avocado defect occurs as a result of rough handling, impact, or stress during transportation and storage. the initial symptoms of this defect have tan, scaly, and bruised skin lines that are quite severe throughout the area of the fruit that will not change color as it ripens, especially when the fruit is still small. other symptoms experienced by this defect can be caused by friction or insect factors that make the outer layer of the fruit skin severely damaged.")

            col5, col6 = st.columns([1, 2])
            with col5:
                st.image("assets/blackorbrown.jpg", caption="Black or brown spots Avocado Image")
            with col6:
                st.subheader("Black or Brown Spots")
                st.write("Black or brown spots occurring on avocado fruits are caused by the influence of cold Air temperatures and wet skin external surface conditions during or before harvest characterized by shiny dark brown to black lesions or marks on the avocado skin, most often on the surface or near the stem end of the fruit. this symptom often occurs on both hard and soft fruit, although receiving agents usually observe the symptom on fruit delivered in hard condition. another cause of the occurrence of this defect is due to the process of transporting and packing the fruit at improper low temperatures and is usually more susceptible to being affected on larger avocados after harvest than smaller fruits.")
            

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass