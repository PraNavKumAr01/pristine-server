import onnxruntime as ort
from PIL import Image
import clip
from torchvision import transforms

def preprocess_image_clip(image_path):

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to('cpu')
    return image

model_image_quant = "./models/clip-image-encoder-quant-int8.ort"
model_text_quant = "./models/clip-text-encoder-quant-int8.ort"

def encode_image(img_path):
    ort_session = ort.InferenceSession(model_image_quant)
    input_name = ort_session.get_inputs()[0].name
    
    image_input = preprocess_image_clip(img_path)

    outputs = ort_session.run(None, {input_name: image_input.numpy()})
    return outputs[0]

def encode_text(text):
    ort_session = ort.InferenceSession(model_text_quant)
    input_name = ort_session.get_inputs()[0].name

    token_input = clip.tokenize([text]).to("cpu")
    
    outputs = ort_session.run(None, {input_name: token_input.numpy()})
    return outputs[0]

image_path = "/Users/pranav/Documents/Projects/pristine-server/experiments/pexels-photo-1108099.jpeg"
text = "a dog"

image_features = encode_image(image_path)
text_features = encode_text(text)

print("Image features:", image_features)
print("Text features:", text_features)
