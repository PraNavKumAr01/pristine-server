import os
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPHandler:
    def __init__(self, model_dir=None):
        """
        Initialize BLIP model handler with local storage capability
        Args:
            model_dir: Directory to store the model files. If None, uses CWD/models/blip-caption
        """
        if model_dir is None:
            model_dir = str(Path.cwd() / 'models' / 'blip-caption')
        self.model_dir = Path(model_dir)
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = None
        self.model = None
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.load_or_download_model()

    def load_or_download_model(self):
        """Load model from local storage or download if not available"""
        model_path = self.model_dir / 'model'
        processor_path = self.model_dir / 'processor'

        try:
            if model_path.exists() and processor_path.exists():
                print("Loading BLIP model from local storage...")
                self.processor = BlipProcessor.from_pretrained(str(processor_path))
                self.model = BlipForConditionalGeneration.from_pretrained(str(model_path))
            else:
                print("Downloading BLIP model...")
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
                
                print("Saving BLIP model to local storage...")
                self.processor.save_pretrained(str(processor_path))
                self.model.save_pretrained(str(model_path))
                
            print("✓ BLIP model ready!")
            
        except Exception as e:
            print(f"✗ Error loading/downloading BLIP model: {e}")
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