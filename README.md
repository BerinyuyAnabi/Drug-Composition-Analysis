**Methodology**
Dataset Acquisition: We obtained a dataset of drugs from various medical sources and converted it into a CSV file for development. Additionally, we scraped data from Drugs.com to include drug links in our dataset.
Image Processing and Text Extraction: We employed EasyOCR, a Python library that uses deep learning models for Optical Character Recognition (OCR). EasyOCR supports over 80 languages and uses convolutional and recurrent neural networks to extract text from images.
Text Cleaning and Matching: The extracted text is cleaned and split to ensure efficient comparison. Units (mg, mL, units) are combined with the preceding words. This cleaned data is then compared with the dataset's active ingredients and drug strengths.
Data Processing: The dataset is processed to handle missing values and prepare it for comparison.
Matching and Verification: The application performs a double-layer comparison: first, it identifies matching active ingredients, and second, it verifies the drug strength against the extracted text. 
Matching results are used to confirm the drug's legitimacy. Users can input drug names for verification, and if a match is found, they receive a confirmation message along with a list of similar drugs and relevant links.


**Steps to host the application on your local server**
Open Terminal or Command Prompt.
Set Up a Virtual Environment (Optional): It is recommended to create a virtual environment for your project:
Create the virtual environment.
Activate it.
Install Required Libraries: Use a package manager like pip to install the necessary libraries.
Run the Streamlit Application: Start your application using the appropriate command with the name of your Python file ( streamlit run app.py)
Access the Application: Open a web browser and navigate to the provided local URL (usually http://localhost:8501).
Interact with the Application: Use the application features to upload images or capture them and analyze the extracted text.
**Troubleshooting**
Check for error messages in the terminal if you encounter issues.

Link to Video:
https://youtu.be/yZNNp2UEVq4 
