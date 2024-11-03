from sentence_transformers import SentenceTransformer
from pathlib import Path
from pathlib import Path

def is_folder_empty(folder_path):
    return not any(Path(folder_path).iterdir())

class ClipModelHandler:
    def __init__(self, model_dir=None):
        """
        Initialize SentenceTransformer model handler with local storage capability
        Args:
            model_dir: Directory to store the model files. If None, uses CWD/models/clip-model
        """
        if model_dir is None:
            model_dir = str(Path.cwd() / 'models' / 'clip-model')
        self.model_dir = Path(model_dir)
        self.model_name = "clip-ViT-B-16"
        self.model = None
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.load_or_download_model()

    def load_or_download_model(self):
        """Load model from local storage or download if not available"""
        model_path = self.model_dir
        print(f"Model path: {model_path}")
        
        try:
            if not is_folder_empty(model_path):
                print("Loading Clip model from local storage...")
                self.model = SentenceTransformer(
                    str(self.model_dir),
                )
            else:
                print("Downloading Clip model...")
                self.model = SentenceTransformer(
                    self.model_name,
                )
                
                print("Saving Clip model to local storage...")
                self.model.save(str(self.model_dir))
                
            print("✓ Clip model ready!")
            
        except Exception as e:
            print(f"✗ Error loading/downloading Clip model: {e}")
            raise

    def get_model(self):
        """
        Get the loaded SentenceTransformer model
        Returns:
            SentenceTransformer: Loaded model instance
        """
        if self.model is None:
            print("Model is not loaded.")
        return self.model