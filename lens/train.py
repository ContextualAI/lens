from model import Lens, LensDataset, LensProcessor
import requests
from lens import Lens, LensProcessor
from PIL import Image
import torch

img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "What is the image about?"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_loss(lens, samples):
    total_loss = 0
    for module in ["tags", "attributes"]:
        image_features = lens.clip_model.encode_image(
            samples["clip_image"].to(device)
        )
        text_features = lens.clip_model.encode_text(
            samples["questions"].to(device)
        )

def main():
    lens = Lens()
    processor = LensProcessor()
    with torch.no_grad():
        samples = processor([raw_image],[question])
        output = lens(samples)
    calc_loss(lens, samples)



