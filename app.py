# streamlit_app.py

import streamlit as st
from PIL import Image
from docx import Document
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to generate text from a given image region
def generate_text(image_region):
    # Load TrOCR processor
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Load TrOCR model
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    # Preprocess the image region
    pixel_values = processor(images=image_region, return_tensors="pt").pixel_values

    # Generate text from the image region
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# Function to process image and extract text
def process_image(image):
    # Perform text detection using EasyOCR
    reader = easyocr.Reader(['en'])  # Language selection: 'en' for English
    result = reader.readtext(image)

    # Variable to store generated text
    generated_texts = []

    for detection in result:
        bbox = detection[0]
        text = detection[1]

        # Extract the text region
        text_region = image.crop((bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]))

        # Generate text for the text region
        generated_text = generate_text(text_region)
        generated_texts.append(generated_text)

    return generated_texts

# Visualize the detected text regions
def visualize_text_detection(image, result):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for detection in result:
        bbox = detection[0]
        text = detection[1]

        # Create a rectangle patch
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add the text as a label
        plt.text(bbox[0][0], bbox[0][1], text, color='r', fontsize=8)

    plt.axis('off')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Handwriting Detection with Streamlit")

    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the uploaded image
        generated_texts = process_image(image)

        # Display the detected and generated texts
        st.subheader("Detected Texts:")
        for text in generated_texts:
            st.write(text)

        # Save the generated texts to a Word document
        if st.button("Save to Word Document"):
            doc_path = 'Generated_Texts.docx'
            doc = Document()
            for text in generated_texts:
                doc.add_paragraph(text)
            doc.save(doc_path)
            st.success(f"Generated texts saved to {doc_path}")

        # Visualize the detected text regions
        visualize_text_detection(image, result)

if __name__ == "__main__":
    main()
