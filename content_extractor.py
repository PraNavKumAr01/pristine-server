import base64
from docx import Document
import fitz
from PIL import Image
import io
import pandas as pd
from image_ocr import SmartOCR
from blip_caption import BLIPHandler
from file_types import image_exts, doc_exts, pdf_exts, excel_exts, csv_exts, get_file_extension

def extract_image_content(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_doc_content(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    ocr_handler = SmartOCR()
    blip_handler = BLIPHandler()

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text from the page
        page_text = page.get_text()
        full_text += page_text + " "

        # Extract and process images
        page_images = page.get_images(full=True)

        for img_index, img in enumerate(page_images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            if "mask" in base_image:
                mask_bytes = base_image["mask"]
                pix = fitz.Pixmap(fitz.csRGB, image_bytes, mask_bytes)
            else:
                pix = fitz.Pixmap(image_bytes)

            img_data = pix.tobytes("png")
            img_buffer = io.BytesIO(img_data)
            pil_image = Image.open(img_buffer)

            # Perform OCR on the image
            ocr_text = ocr_handler.process_image(pil_image)
            if ocr_text:
                full_text += ocr_text + " "
            image_caption = blip_handler.generate_caption(pil_image)
            full_text += image_caption

            pix = None  # Free up memory

    doc.close()
    return full_text.strip()

def extract_excel_content(file_path):
    df = pd.read_excel(file_path)
    return df.to_string(index=False)

def extract_csv_content(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def extract_text_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def extract_content(file_path):
    ext = get_file_extension(file_path)
    if ext in image_exts:
        return extract_image_content(file_path)
    elif ext in doc_exts:
        return extract_doc_content(file_path)
    elif ext in pdf_exts:
        return extract_pdf_content(file_path)
    elif ext in excel_exts:
        return extract_excel_content(file_path)
    elif ext in csv_exts:
        return extract_csv_content(file_path)
    else:
        return extract_text_content(file_path)