from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import evalulate_model.ai as model

page = st.sidebar.radio(f"Page", ["Prediction", "Analysis", "About"])
if page == "Prediction":
    with st.spinner("Loading YOLO"):
        net = model.get_yolo_net("yolov4-tiny-custom.cfg", "yolov4-tiny-custom_final.weights")

    CLASSES = []
    with open("classes.txt") as f:
        for line in f.read().split("\n"):
            CLASSES.append(line)
    COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")
    picture = st.camera_input("Take a picture")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "WEBP"])

    bytes_to_use = picture if picture is not None else uploaded_image
    use_preset = st.checkbox("Use default image?")
    if use_preset is True:
        preset_img = st.selectbox("Please select image", ["Graphics Card", "Power Supply"])
        if preset_img == "Graphics Card":
            img = cv2.imread("graphiccard.jpg")
        elif preset_img == "Power Supply":
            img = cv2.imread("oiwersupply.jpg")
    else:
        preset_img = None
    # bytes_to_use = img = cv2.imread('dog.jpeg')
    if bytes_to_use is not None and preset_img is None:
        with st.spinner("Loading Image"):
            bytes_to_use = bytes_to_use.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_to_use, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            width, height, _ = image.shape

        classid, output_classes, boxes, confidence = model.yolo_forward(net, CLASSES, image, 0.25, save_image=False)
        with st.expander("Open me"):
            st.write(list(x for x in zip(classid, output_classes, boxes, confidence)))
        model.yolo_show_img(image, classid, boxes, output_classes, confidence, COLORS)
        st.image(image)
    elif preset_img is not None:
        classid, output_classes, boxes, confidence = model.yolo_forward(net, CLASSES, img, 0.25, save_image=False)
        with st.expander("Open me"):
            st.write(
                list(
                    {
                        "Class Index": cls,
                        "Class Name": out_cls,
                        "Bounding Boxes": box,
                        "Confidence": f"{conf:.3f}",
                    }
                    for cls, out_cls, box, conf in zip(classid, output_classes, boxes, confidence)
                )
            )
        model.yolo_show_img(img, classid, boxes, output_classes, confidence, COLORS)
        st.image(img)
elif page == "Analysis":
    matrix_type = st.selectbox("Do Normalize?", ["Normalized Confusion Matrix", "Confusion Matrix"])
    if matrix_type == "Normalized Confusion Matrix":
        image = Image.open("normalized_confusion_matrix.jpg")
        st.image(image)
    elif matrix_type == "Confusion Matrix":
        image = Image.open("confusion_matrix.jpg")
        st.image(image)
elif page == "About":
    readme = Path("README.md")
    st.write(readme.read_text())
