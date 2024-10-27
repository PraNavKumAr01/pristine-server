from pathlib import Path
from huggingface_hub import hf_hub_download

class CLIPModelHandler:
    def __init__(self, model_dir=None):
        """
        Initialize CLIP model handler with local storage capability
        Args:
            model_dir: Directory to store the model files. If None, uses CWD/models/clip-onnx-int8
        """
        if model_dir is None:
            model_dir = Path.cwd() / 'models' / 'clip-onnx-int8'
        self.model_dir = Path(model_dir)
        self.model_repo = "PK03/clip-onnx-int8"
        self.files_to_download = [
            "clip-image-encoder-quant-int8.ort",
            "clip-image-encoder-quant-int8.required_operators.config",
            "clip-image-encoder-quant-int8.required_operators.with_runtime_opt.config",
            "clip-text-encoder-quant-int8.ort",
            "clip-text-encoder-quant-int8.required_operators.config",
            "clip-text-encoder-quant-int8.required_operators.with_runtime_opt.config"
        ]
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.download_or_load_model()
    
    def download_or_load_model(self):
        """Download each model file if not already in the local storage"""
        for file_name in self.files_to_download:
            local_path = self.model_dir / file_name
            if not local_path.exists():
                print(f"Downloading {file_name} from Hugging Face Hub...")
                try:
                    hf_hub_download(
                        repo_id=self.model_repo,
                        filename=file_name,
                        local_dir=self.model_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"✓ {file_name} downloaded successfully!")
                except Exception as e:
                    print(f"✗ Error downloading {file_name}: {e}")
            else:
                print(f"✓ {file_name} already exists locally.")
    
    def get_model_paths(self):
        """
        Returns the paths to the image and text encoder models.
        Returns:
            tuple: Paths to image and text encoder models
        """
        image_model_path = self.model_dir / "clip-image-encoder-quant-int8.ort"
        text_model_path = self.model_dir / "clip-text-encoder-quant-int8.ort"
        
        return str(image_model_path), str(text_model_path)