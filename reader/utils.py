import io
import fitz
import pytesseract
from PIL import Image

def is_scan_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    scan_page_count = 0
    
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text()
        images = page.get_images(full=True)
        
        if images and (not text or len(text.strip()) < 100):
            scan_page_count += 1
    doc.close()
    
    return scan_page_count / total_pages > 0.5 if total_pages > 0 else False

def split_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        
        img = Image.open(io.BytesIO(pix.tobytes()))
        images.append(img)

    pdf_document.close()
    return images

def read_scanPDF(pdf_path, lang='vie', 
                                tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images = split_pdf(pdf_path)
    
    extracted_texts = []
    
    for img in images:
        text = pytesseract.image_to_string(
            img, 
            lang=lang,
            config='--psm 1 --oem 3'  
        )
        extracted_texts.append(text)
    
    return extracted_texts