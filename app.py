import streamlit as st
import easyocr
import cv2
import pandas as pd
import numpy as np
import re
from PIL import Image
import logging
import base64

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

def extract_text(image):
    """Extracts text from an image using EasyOCR."""
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image)
        full_text = " ".join([detection[1] for detection in result])
        return full_text
    except Exception as e:
        st.error(f"Error in text extraction: {e}")
        logging.error(f"Error in text extraction: {e}")
        return None

def clean_and_split_list(input_list):
    pattern = r'[\W_]+'
    cleaned_list = [w for word in input_list for w in re.split(pattern, word) if w]
    final_list = []
    i = 0
    while i < len(cleaned_list):
        if i < len(cleaned_list) - 1 and cleaned_list[i + 1] in ['mg', 'mL', 'units']:
            final_list.append(cleaned_list[i] + cleaned_list[i + 1])
            i += 2
        else:
            final_list.append(cleaned_list[i])
            i += 1
    return final_list

def process_item(item):
    if not isinstance(item, str):
        return item
    item = re.sub(r'\*\*.*?\*\*', '', item).strip()
    item = item.lower()
    return item

def clean_data(input_list):
    processed_list = [[process_item(item) for item in sublist] for sublist in input_list]
    return processed_list

@st.cache
def load_data():
    return pd.read_csv("updated_file.csv")

def image_from_base64(base64_str):
    """Convert base64 string to image."""
    image_data = base64.b64decode(base64_str)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

st.title('Drug Composition Analysis')

# Option to upload an image or use webcam
option = st.selectbox("Choose an option:", ["Upload an image", "Capture with webcam"])

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img.save('uploaded_image.jpg')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        image_path = 'uploaded_image.jpg'

elif option == "Capture with webcam":
    st.write("Click the button below to take a picture.")
    captured_image = st.camera_input("Take a picture")
    if captured_image:
        # Save the captured image
        img = Image.open(captured_image)
        img.save('captured_image.jpg')
        st.image(img, caption='Captured Image', use_column_width=True)
        image_path = 'captured_image.jpg'

# Process the image if one is available
if 'image_path' in locals():
    full_text = extract_text(image_path)
    st.write("Extracted Text:")
    st.write(full_text)

    full_text_list = full_text.strip().split()
    cleaned_list = clean_and_split_list(full_text_list)
    cleaned_list = [word.lower() for word in cleaned_list]

    st.write("Cleaned List:")
    st.write(cleaned_list)

    medicine_data = load_data()

    numeric_data = medicine_data.select_dtypes(include=np.number)
    non_numeric = medicine_data.select_dtypes(include=['object'])

    for col in non_numeric:
        most_frequent_value = non_numeric[col].mode()[0]
        non_numeric[col].fillna(most_frequent_value, inplace=True)

    medicine_data = pd.concat([numeric_data, non_numeric], axis=1)

    active_substance = medicine_data[['ActiveIngredient', 'Strength']]
    active_substance = active_substance.values.tolist()

    processed_list = clean_data(active_substance)

    matching_words = []
    matching_indices = []

    for sublist1 in cleaned_list:
        for index, sublist2 in enumerate(processed_list):
            if any(sublist1.lower() in word2.lower() for word2 in sublist2):
                matching_words.append(sublist1)
                matching_indices.append(index)

    st.write("Matching words:", set(matching_words))

    name_of_medicine = medicine_data['DrugName']
    drug_urls = medicine_data['DrugLink']
    medicine_names = name_of_medicine.iloc[matching_indices].unique()
    medicine_names = [name.lower() for name in medicine_names]

    user_input = st.text_input("Enter a name of the drug:").lower()
    if user_input:
        if user_input in medicine_names:
            st.write('')
        else:
            st.write(f"{user_input} not found in the list.")

        medicine_name = [name for name in medicine_names if name != user_input]
        st.write("This list of drugs that also have similar active substances as components are:")
        for name in medicine_name[:10]:
            drug_url = medicine_data[medicine_data['DrugName'].str.lower() == name]['DrugLink'].values[0]
            st.markdown(f"[{name}]({drug_url})")

        if user_input in medicine_names:
            st.write("The drug is legitimate")
            drug_link = drug_urls[name_of_medicine.str.lower() == user_input].values[0]
            st.write(f"Drug Link: [{user_input}]({drug_link})")
        else:
            st.write("No data about this drug, take caution!!")
