from model import Lens, LensDataset, LensProcessor
import requests
from lens import Lens, LensProcessor
from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
#model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
lens = Lens()
model = lens.clip_model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilitie
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
