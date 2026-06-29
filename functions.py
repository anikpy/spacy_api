import numpy as np
import pytesseract
from PIL import Image
import cv2
from pdf2image import convert_from_path
from docx import Document
import pandas as pd
import io


def ocr_from_image(image: Image.Image) -> str:
    # Convert PIL Image to OpenCV image
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    return text

def ocr_from_pdf(pdf_file: io.BytesIO) -> str:
    images = convert_from_path(pdf_file)
    extracted_text = ""
    for i, image in enumerate(images):
        extracted_text += f"Page {i + 1}:\n{ocr_from_image(image)}\n\n"
    return extracted_text

def ocr_from_word(docx_file: io.BytesIO) -> str:
    doc = Document(docx_file)
    extracted_text = ""
    for para in doc.paragraphs:
        extracted_text += para.text + "\n"
    return extracted_text

def ocr_from_excel(xlsx_file: io.BytesIO) -> str:
    df = pd.read_excel(xlsx_file)
    extracted_text = ""
    for column in df.columns:
        extracted_text += df[column].astype(str).str.cat(sep='\n') + "\n"
    return extracted_text

