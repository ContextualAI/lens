'''
https://dienhoa.github.io/dhblog/posts/finetune_clip.html
https://huggingface.co/docs/transformers/training
'''

from model import Lens, LensDataset, LensProcessor
import requests
from lens import Lens, LensProcessor
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer
import torch
import clip
import numpy as np

img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "What is the image about?"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LensTrainer():
    def __init__(self):
        self.model = Lens()
        self.processor = LensProcessor()
        with torch.no_grad():
            inputs = self.processor([raw_image],[question])
        self.compute_loss(self.model, inputs)

    def compute_loss(model, inputs):
        total_loss = 0
        lm_likelihood = torch.ones((5, 1))
        for desc in ["tags", "attributes"]:
            image_features = model.clip_model.encode_image(
                inputs["clip_image"].to(device)
            )
            text_features = model.clip_model.encode_text(
                clip.tokenize(inputs[desc][0]).to(device)
            )
            logits = text_features @ image_features.T 
            desc_likelihood = logits.softmax(dim=0).to(device).numpy()
            total_loss += rel_entr(lm_likelihood, desc_likelihood)

def main():
    trainer = LensTrainer()



