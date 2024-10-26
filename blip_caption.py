import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPHandler:
    def __init__(self, model_dir='experiments/models/blip'):
        """
        Initialize BLIP model handler with local storage capability
        Args:
            model_dir: Directory to store the model files
        """
        self.model_dir = model_dir
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = None
        self.model = None
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.load_or_download_model()

    def load_or_download_model(self):
        """Load model from local storage or download if not available"""
        model_path = os.path.join(self.model_dir, 'model')
        processor_path = os.path.join(self.model_dir, 'processor')

        try:
            # Try to load from local storage
            if os.path.exists(model_path) and os.path.exists(processor_path):
                print("Loading BLIP model from local storage...")
                self.processor = BlipProcessor.from_pretrained(processor_path)
                self.model = BlipForConditionalGeneration.from_pretrained(model_path)
            else:
                print("Downloading BLIP model...")
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
                
                print("Saving BLIP model to local storage...")
                self.processor.save_pretrained(processor_path)
                self.model.save_pretrained(model_path)
                
            print("BLIP model ready!")
            
        except Exception as e:
            print(f"Error loading/downloading BLIP model: {e}")
            raise

    def generate_caption(self, image_path):
        """
        Generate caption for an image
        Args:
            image_path: Path to the image file
        Returns:
            str: Generated caption
        """
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt")
            
            self.model.eval()
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=30, min_length=20)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ""