import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import Optional, Tuple
from PIL import Image

class SmartOCR:
    def __init__(self, lang: str = 'en'):
        """
        Initialize the SmartOCR class with specified language.

        Args:
            lang (str): Language code for OCR (default: 'en')
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def _check_for_text(self, image: np.ndarray, min_text_area_ratio: float = 0.05) -> Tuple[bool, np.ndarray]:
        """
        Check if the image contains text using basic image processing techniques.

        Args:
            image (np.ndarray): Image as a numpy array
            min_text_area_ratio (float): Minimum ratio of text-like areas to image area

        Returns:
            Tuple[bool, np.ndarray]: (True if text likely exists, processed image)
        """
        try:
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
            return False, image

    def process_image(self, pil_image: Image.Image) -> Optional[str]:
        """
        Process an image with text detection and OCR.

        Args:
            pil_image (PIL.Image.Image): PIL Image object

        Returns:
            Optional[str]: Extracted text if found, None otherwise
        """
        try:
            print("Processing")
            # Convert PIL Image to numpy array
            image = np.array(pil_image)

            # Check for text presence
            has_text, _ = self._check_for_text(image)
            print(has_text)

            if not has_text:
                print("No significant text detected in the image", flush=True)
                return None

            # Perform OCR
            result = self.ocr.ocr(image, cls=True)

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