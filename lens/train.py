'''
https://dienhoa.github.io/dhblog/posts/finetune_clip.html
https://huggingface.co/docs/transformers/training
'''

from model import Lens, LensDataset, LensProcessor
import requests
from lens import Lens, LensProcessor
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, CLIPProcessor, CLIPModel
import torch
import numpy as np

img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "What is the image about?"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LensTrainer():
    def __init__(self):
        self.model = Lens()
        self.processor = LensProcessor()

    def compute_loss(model, inputs):
        total_loss = 0
        lm_likelihood = torch.ones((5, 1))
        for desc in ["tags", "attributes"]:
            import pdb; pdb.set_trace()

def main():
    lensTrainer = LensTrainer()
    model = lensTrainer.model
    inputs = model.processor([raw_image],[question])
    lensTrainer.compute_loss(model, inputs)
    trainer = LensTrainer()