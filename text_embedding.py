import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

class SentenceTransformerHandler:
    def __init__(self, model_dir=None):
        """
        Initialize SentenceTransformer model handler with local storage capability
        Args:
            model_dir: Directory to store the model files. If None, uses CWD/models/text-onnx-int8
        """
        if model_dir is None:
            model_dir = str(Path.cwd() / 'models' / 'text-onnx-int8')
        self.model_dir = Path(model_dir)
        self.model_name = "all-MiniLM-L6-v2"
        self.onnx_version = "model_qint8_avx512_vnni.onnx"
        self.model = None
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.load_or_download_model()

    def load_or_download_model(self):
        """Load model from local storage or download if not available"""
        model_path = self.model_dir / self.onnx_version
        print(f"Model path: {model_path}")
        
        try:
            if model_path.exists():
                print("Loading SentenceTransformer model from local storage...")
                self.model = SentenceTransformer(
                    str(self.model_dir),
                    backend="onnx",
                    model_kwargs={
                        "file_name": "model_qint8_avx512_vnni.onnx",
                        "provider": "CPUExecutionProvider"
                    }
                )
            else:
                print("Downloading SentenceTransformer model...")
                self.model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                    model_kwargs={
                        "file_name": "onnx/model_qint8_avx512_vnni.onnx",
                        "provider": "CPUExecutionProvider"
                    }
                )
                
                print("Saving SentenceTransformer model to local storage...")
                self.model.save(str(self.model_dir))
                
            print("✓ SentenceTransformer model ready!")
            
        except Exception as e:
            print(f"✗ Error loading/downloading SentenceTransformer model: {e}")
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