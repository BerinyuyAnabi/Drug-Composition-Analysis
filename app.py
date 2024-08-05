import streamlit as st
import easyocr
import cv2
import numpy as np
import re
from PIL import Image
from difflib import SequenceMatcher
import pickle


# Function to extract text from an image using EasyOCR
def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)

    # Display the image with bounding boxes around detected text
    image = cv2.imread(image_path)
    full_text = ""
    for detection in result:
        top_left = tuple(int(coord) for coord in detection[0][0])
        bottom_right = tuple(int(coord) for coord in detection[0][2])
        text = detection[1]
        full_text += text + " "
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
        image = cv2.putText(image, text, top_left, font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    return result, full_text.strip()


# Function to clean and split the information from the composition section on drugs
def clean_and_split_list(input_list):
    pattern = r'[\W_0-9]+'
    cleaned_list = []
    for word in input_list:
        parts = re.split(pattern, word)
        cleaned_list.extend(parts)
    cleaned_list = [w.lower() for w in cleaned_list if w]
    return cleaned_list


# Function to calculate the similarity between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Main function to run the Streamlit app
def main():
    st.title("Drug Composition Text Extraction and Matching")

    # Load the processed data
    with open("processed_active_substances.pkl", "rb") as f:
        processed_list = pickle.load(f)


    with open("medicine_names.pkl", "rb") as f:
        medicine_names = pickle.load(f)

    with open("inn_common_names.pkl", "rb") as f:
        inn_common_names = pickle.load(f)

    with open("pharmacotherapeutic_groups.pkl", "rb") as f:
        pharmacotherapeutic_groups = pickle.load(f)

    with open("links.pkl", "rb") as f:
        links = pickle.load(f)

    # Option to upload image or take a photo
    option = st.radio("Choose input method:", ("Upload an image", "Take a photo with camera"))

    if option == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo...")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.convert('RGB')
        img_path = "uploaded_image.jpg"
        img.save(img_path)

        # Extract text from image
        result, full_text = extract_text(img_path)
        st.write("Extracted Text in Original Form:")
        st.write(full_text)

        # Process the full text
        full_text_list = full_text.strip().split()
        cleaned_list = clean_and_split_list(full_text_list)
        st.write("Cleaned List:")
        st.write(cleaned_list)

        # Matching words and finding indices
        matching_indices = []
        for word in cleaned_list:
            for index, sublist in enumerate(processed_list):
                if word in sublist:
                    matching_indices.append(index)

        st.write("Indices of matching sublists:")
        st.write(set(matching_indices))

        # Extract matching medicine names and additional information
        if matching_indices:
            st.write("Matching Information:")
            for index in set(matching_indices):
                st.write(f"Medicine Name: {medicine_names[index]}")
                st.write(f"INN/Common Name: {inn_common_names[index]}")
                st.write(f"Pharmacotherapeutic Group: {pharmacotherapeutic_groups[index]}")
                st.write(f"[More Information]({links[index]})")
        else:
            st.write("No matching active substances found in the provided image.")


if __name__ == "__main__":
    main()
