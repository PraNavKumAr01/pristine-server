import onnxruntime as ort
from PIL import Image
import clip
from torchvision import transforms
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings

class CLIPEmbeddings(Embeddings):
    def __init__(
        self,
        image_model_path: str,
        text_model_path: str
    ):
        self.image_model_path = image_model_path
        self.text_model_path = text_model_path
        self.image_session = ort.InferenceSession(image_model_path)
        self.text_session = ort.InferenceSession(text_model_path)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to('cpu')
        return image.numpy()

    def embed_image(self, image_path: str) -> List[float]:
        input_name = self.image_session.get_inputs()[0].name
        image_input = self._preprocess_image(image_path)
        outputs = self.image_session.run(None, {input_name: image_input})
        return outputs[0][0].tolist()

    def embed_text(self, text: str) -> List[float]:
        input_name = self.text_session.get_inputs()[0].name
        token_input = clip.tokenize([text]).to("cpu")
        outputs = self.text_session.run(None, {input_name: token_input.numpy()})
        return outputs[0][0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the text encoder"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the text encoder"""
        return self.embed_text(text)
