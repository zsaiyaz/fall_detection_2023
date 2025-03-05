import os
import torch
from PIL import Image
import streamlit as st
from helper import load_model, play_stored_video, play_webcam
import settings
from pathlib import Path

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Fall Detection with YOLOv8', page_icon='logo\\fall2.png', layout='wide', initial_sidebar_state='expanded')

def main():
    # Sidebar
    # st.title("Fall Detection using YOLOv8",style='centered')    
    st.markdown("<h1 style='text-align: center; color: black; '>FALL DETECTION </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;font-style: italic;'> ----USING YOLOv8----</h2>", unsafe_allow_html=True)

    st.sidebar.header("ML Model Config")

    # Model Options
    # model_type = st.sidebar.radio(
    #     "Select Task", ['Detection'])
    model_type = 'Detection'

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

    if model_type == 'Detection':
        dirpath_locator = settings.DETECT_LOCATOR
        # model_path = Path(settings.DETECTION_MODEL)

        # Let the user select the model type
        model_type = st.sidebar.selectbox("Select Model Type", list(settings.MODEL_DICT.keys()), index=0)

        # Get the corresponding model path from the MODEL_DICT
        model_path = settings.MODEL_DICT[model_type]

    # Load model
    try:
        model = load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    # Image/Video Config
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        # save_radio = st.sidebar.radio("Save image to download", ["Yes", "No"])
        # save = True if save_radio == 'Yes' else False
        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    image = Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image", use_container_width=True)
                else:
                    image = Image.open(source_img)
                    st.image(image, caption="Uploaded Image")           
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)
        
        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                     use_container_width=True)
            else:
                if st.sidebar.button('Detect'):
                    with torch.no_grad():
                        res = model.predict(image,
                                        save=False,
                                        save_txt=False,
                                        exist_ok=True,
                                        conf=confidence
                                        )
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image',use_container_width=True)
                        # IMAGE_DOWNLOAD_PATH = f"runs/{dirpath_locator}/predict/image0.jpg"
                        # with open(IMAGE_DOWNLOAD_PATH, 'rb') as fl:
                        #     st.download_button("Download object-detected image",
                        #                    data=fl,
                        #                    file_name="image0.jpg",
                        #                    mime='image/jpg'
                        #                    )
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")               

    elif source_radio == settings.VIDEO:
        # Play stored video
        play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        # Play webcam
        play_webcam(confidence, model)

    else:
        st.error("Please select a valid source type!")


if __name__ == "__main__":
    main()
