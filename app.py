import streamlit as st
from PIL import Image, ImageDraw
from docx import Document
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Load TrOCR processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

# Load TrOCR model
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Function to generate text from a given image region
def generate_text(image_region):
    # Preprocess the image region
    pixel_values = processor(images=image_region, return_tensors="pt").pixel_values

    # Generate text from the image region
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# Function to process image and extract text
def process_image(image):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Perform text detection using EasyOCR
    reader = easyocr.Reader(['en'])  # Language selection: 'en' for English
    result = reader.readtext(image_np)

    # Variable to store generated text and bounding boxes
    generated_texts = []
    bounding_boxes = []

    for detection in result:
        bbox = detection[0]
        text = detection[1]

        # Extract the text region
        text_region = image.crop((bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]))

        # Generate text for the text region
        generated_text = generate_text(text_region)
        generated_texts.append(generated_text)
        bounding_boxes.append(bbox)

    return generated_texts, bounding_boxes

# Visualize the detected text regions overlaid on the image
def visualize_text_detection(image, generated_texts, bounding_boxes):
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    
    for bbox, generated_text in zip(bounding_boxes, generated_texts):
        # Create a rectangle patch
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add the generated text as a label
        plt.text(bbox[0][0], bbox[0][1], generated_text, color='r', fontsize=8)

    # Display the image with overlaid text
    st.image(image, caption="Detected Texts", use_column_width=True)

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
        generated_texts, bounding_boxes = process_image(image)

        # Display the detected and generated texts
        st.subheader("Generated Texts:")
        st.write(' '.join(generated_texts))

        # Save the generated texts to a Word document
        if st.button("Save to Word Document"):
            doc_path = 'Generated_Texts.docx'
            doc = Document()
            doc.add_paragraph(' '.join(generated_texts))
            doc.save(doc_path)
            st.success(f"Generated texts saved to {doc_path}")

        # Visualize the detected text regions
        visualize_text_detection(image, generated_texts, bounding_boxes)

if __name__ == "__main__":
    main()
