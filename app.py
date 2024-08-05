import streamlit as st
import easyocr
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import re
import pandas as pd
import numpy as np
import base64
from io import BytesIO

# Define functions here

def extract_text(image_path):
    """Extracts text from an image using EasyOCR."""
    try:
        reader = easyocr.Reader(['en'])  # Specify the language(s) you want to use
        result = reader.readtext(image_path)

        # Load image with OpenCV
        image = cv2.imread(image_path)
        full_text = ""  # Initialize an empty string to hold the full text
        for detection in result:
            top_left = tuple(int(coord) for coord in detection[0][0])  # Convert coordinates to integers
            bottom_right = tuple(int(coord) for coord in detection[0][2])  # Convert coordinates to integers
            text = detection[1]
            full_text += text + " "  # Append the detected text to the full_text string
            font = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
            image = cv2.putText(image, text, top_left, font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert image to displayable format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        return result, full_text.strip(), image_bytes
    except Exception as e:
        st.error(f"Error in text extraction: {e}")
        return None, None, None

def clean_and_split_list(input_list):
    pattern = r'[\W_]+'  # Match any non-word character or underscore
    cleaned_list = []
    for word in input_list:
        parts = re.split(pattern, word)
        cleaned_list.extend(parts)
    cleaned_list = [w for w in cleaned_list if w]

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
    processed_list = []
    for sublist in input_list:
        cleaned_sublist = [process_item(item) for item in sublist]
        processed_list.append(cleaned_sublist)
    return processed_list

# Load your CSV file
@st.cache
def load_data():
    return pd.read_csv("updated_file.csv")

# Streamlit UI
st.title('Drug Composition Analysis')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img.save('uploaded_image.jpg')
        st.image(img, caption='Uploaded Image', use_column_width=True)

        result, full_text, image_bytes = extract_text('uploaded_image.jpg')
        if result is not None and full_text is not None and image_bytes is not None:
            st.image(image_bytes, caption='Processed Image', use_column_width=True)

            st.write("Extracted Text:")
            st.write(full_text)

            full_text_list = full_text.strip().split()
            cleaned_list = clean_and_split_list(full_text_list)
            cleaned_list = [word.lower() for word in cleaned_list]

            st.write("Cleaned List:")
            st.write(cleaned_list)

            # Load and process the data
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

            # Replace 'MedicineName' with the correct column name if different
            name_of_medicine = medicine_data['DrugName']
            drug_urls = medicine_data['DrugLink']  # Assuming this column exists in your CSV
            medicine_names = name_of_medicine.iloc[matching_indices].unique()
            medicine_names = [name.lower() for name in medicine_names]

            user_input = st.text_input("Enter a name of the drug:").lower()
            if user_input:
                if user_input in medicine_names:
                    st.write('')
                else:
                    st.write(f"{user_input} not found in the list.")

                # Limit to first 10 similar drugs
                medicine_name = [name for name in medicine_names if name != user_input]
                st.write("This list of drugs that also have similar active substances as components are:")
                for name in medicine_name[:10]:  # Slicing to get the first 10 items
                    drug_url = medicine_data[medicine_data['DrugName'].str.lower() == name]['DrugLink'].values[0]
                    st.markdown(f"[{name}]({drug_url})")

                if user_input in medicine_names:
                    st.write("The drug is legitimate")
                    # Get and display the drug URL for the input drug name
                    drug_link = drug_urls[name_of_medicine.str.lower() == user_input].values[0]
                    st.write(f"Drug Link: [{user_input}]({drug_link})")
                else:
                    st.write("No data about this drug, take caution!!")
    except Exception as e:
        st.error(f"Error in processing the uploaded file: {e}")
