from pathlib import Path
from clip_model import CLIPModelHandler
from blip_caption import BLIPHandler
from text_embedding import SentenceTransformerHandler

def setup_models():
    """
    Downloads and sets up CLIP, BLIP, and SentenceTransformer models
    in the current working directory under the models directory
    """
    base_dir = Path.cwd() / 'models'
    base_dir.mkdir(exist_ok=True)
    
    model_paths = {}
    
    print("\n=== Setting up CLIP model ===")
    clip_dir = base_dir / 'clip-onnx-int8'
    clip_handler = CLIPModelHandler(model_dir=str(clip_dir))
    image_model_path, text_model_path = clip_handler.get_model_paths()
    model_paths['clip'] = {
        'dir': str(clip_dir),
        'image_model': image_model_path,
        'text_model': text_model_path
    }
    
    print("\n=== Setting up BLIP model ===")
    blip_dir = base_dir / 'blip-caption'
    blip_handler = BLIPHandler(model_dir=str(blip_dir))
    model_paths['blip'] = {
        'dir': str(blip_dir)
    }
    
    print("\n=== Setting up SentenceTransformer model ===")
    st_dir = base_dir / 'text-onnx-int8'
    st_handler = SentenceTransformerHandler(model_dir=str(st_dir))
    model_paths['sentence_transformer'] = {
        'dir': str(st_dir)
    }
    
    print("\nAll models setup complete!")
    print(f"Models directory: {base_dir}")
    print("\nModel locations:")
    for model, paths in model_paths.items():
        print(f"\n{model.upper()}:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
    
    return model_paths

if __name__ == "__main__":
    setup_models()