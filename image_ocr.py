import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import Optional, Tuple

class SmartOCR:
    def __init__(self, lang: str = 'en'):
        """
        Initialize the SmartOCR class with specified language.
        
        Args:
            lang (str): Language code for OCR (default: 'en')
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        
    def _check_for_text(self, image_path: str, min_text_area_ratio: float = 0.05) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if the image contains text using basic image processing techniques.
        
        Args:
            image_path (str): Path to the image
            min_text_area_ratio (float): Minimum ratio of text-like areas to image area
            
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True if text likely exists, processed image)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image at {image_path}", flush=True)
                return False, None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Calculate total image area
            image_area = gray.shape[0] * gray.shape[1]
            
            # Calculate area of potential text regions
            text_area = sum(cv2.contourArea(cnt) for cnt in contours)
            text_ratio = text_area / image_area
            print(text_ratio)
            
            # Check if the ratio exceeds the minimum threshold
            has_text = text_ratio > min_text_area_ratio
            
            return has_text, image

        except Exception as e:
            print(f"Error in text detection: {str(e)}", flush=True)
            return False, None
    
    def process_image(self, image_path: str) -> Optional[str]:
        """
        Process an image with text detection and OCR.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[str]: Extracted text if found, None otherwise
        """
        try:
          print("Processing")
          # Check for text presence
          has_text, image = self._check_for_text(image_path)
          print(has_text)
          
          if not has_text:
              print("No significant text detected in the image", flush=True)
              return None
              
          # Perform OCR
          result = self.ocr.ocr(image_path, cls=True)
          
          if not result or not result[0]:
              return None
          
          # Extract text from results
          extracted_text = " ".join(
              line[1][0] for res in result 
              for line in res if line and line[1]
          )
          
          return extracted_text.strip()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}", flush=True)
            return None

# def demo_ocr():
#     """
#     Demo usage of the SmartOCR class.
#     """
    
#     # Initialize SmartOCR
#     smart_ocr = SmartOCR()
    
#     # Example usage
#     image_path = "/Users/pranav/Documents/Projects/pristine-server/main_content/folder2/uvuygu.jpeg"
#     result = smart_ocr.process_image(image_path)
    
#     if result:
#         print("Extracted text:", flush=True)
#         print(result, flush=True)
#     else:
#         print("No text was detected or extraction failed", flush=True)

# demo_ocr()