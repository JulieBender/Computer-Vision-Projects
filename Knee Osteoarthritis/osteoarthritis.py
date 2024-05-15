import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the YOLO model
model = YOLO('osteoarthritis-s.pt')  # this is a custom tranied model


image_paths = [
    "/Users/juliebender/code/Linkedin posts/grade_0.jpg",
    "/Users/juliebender/code/Linkedin posts/grade_1.jpg",
    "/Users/juliebender/code/Linkedin posts/grade_12.jpg",
    "/Users/juliebender/code/Linkedin posts/grade_3.jpg",
    "/Users/juliebender/code/Linkedin posts/grade_4.jpg",
    "/Users/juliebender/code/Linkedin posts/grade_41.jpg"
    ]

# title and text description
st.title('Osteoarthritis Detection')
st.write('Select an image to detect signs of osteoarthritis using the YOLOv8n model.')

# Image previews and clickable buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.image(image_paths[0], caption="Image 1", width=100)
    if st.button("Select Image 1"):
        selected_image_path = image_paths[0]
with col2:
    st.image(image_paths[1], caption="Image 2", width=100)
    if st.button("Select Image 2"):
        selected_image_path = image_paths[1]
with col3:
    st.image(image_paths[2], caption="Image 3", width=100)
    if st.button("Select Image 3"):
        selected_image_path = image_paths[2]
with col4:
    st.image(image_paths[3], caption="Image 4", width=100)
    if st.button("Select Image 4"):
        selected_image_path = image_paths[3]
with col5:
    st.image(image_paths[4], caption="Image 5", width=100)
    if st.button("Select Image 5"):
        selected_image_path = image_paths[4]
with col6:
    st.image(image_paths[5], caption="Image 6", width=100)
    if st.button("Select Image 6"):
        selected_image_path = image_paths[5]


if 'selected_image_path' in locals():
    selected_image = Image.open(selected_image_path)
    font = ImageFont.load_default()  
    draw = ImageDraw.Draw(selected_image)

    image_array = np.array(selected_image.resize((640, 640)))
    image_batch = [image_array]

    # Run model inference with ultralytics
    results = model(image_batch)

    for result in results:
        detections = result.boxes
        for detection in detections:
            class_id = detection.cls.cpu().numpy().astype(int)
            class_name = model.names[class_id[0]]
            confidence = detection.conf[0].cpu().numpy().astype(float)
            bbox = detection.xyxy.cpu().numpy().astype(int)
            
        
        
            text_position = (0,0)
            text = f"{class_name}, {confidence:.2f}"
            draw.text(text_position, text, fill="red", font=font)

    st.image(selected_image, caption='Detection Results', use_column_width=True)
