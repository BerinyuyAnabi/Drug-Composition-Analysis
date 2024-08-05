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

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
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
        matching_words = []
        matching_indices = []
        for word in cleaned_list:
            for index, sublist in enumerate(processed_list):
                if word in sublist:
                    matching_words.append(word)
                    matching_indices.append(index)

        st.write("Matching words:")
        st.write(set(matching_words))
        st.write("Indices of matching sublists:")
        st.write(set(matching_indices))

        # Check for matching medicine name
        found_match = False
        if len(matching_words) != 0:
            for word in set(cleaned_list):
                if word.lower() in medicine_names:
                    st.write(f"The medicine name is: {word}")
                    found_match = True
                    break
                for item in medicine_names:
                    if word in item:
                        if len(word) > 4:
                            st.write(f"The medicine name is: {item}")
                            found_match = True
                            break
                if found_match:
                    break

            if not found_match:
                max_similarity = 0
                best_match = ""
                for word in set(cleaned_list):
                    for item in medicine_names:
                        similarity = similar(word.lower(), item)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = item
                if max_similarity > 0.7:  # Set a threshold for similarity
                    st.write(f"The most similar medicine name is: {best_match}")
                else:
                    st.write("No matching medicine name found with high similarity.")
        else:
            st.write("No matching medicine name found.")

if __name__ == "__main__":
    main()
