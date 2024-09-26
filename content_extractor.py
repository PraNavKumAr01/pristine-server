import base64
from docx import Document
from PyPDF2 import PdfReader
import pandas as pd
from file_types import image_exts, doc_exts, pdf_exts, excel_exts, csv_exts, get_file_extension

def extract_image_content(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_doc_content(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_pdf_content(file_path):
    pdf = PdfReader(file_path)
    return ''.join([page.extract_text() for page in pdf.pages])

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