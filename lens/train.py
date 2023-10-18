from model import Lens, LensDataset, LensProcessor
import requests
from lens import Lens, LensProcessor
from PIL import Image
import torch
img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "What is the image about?"

lens = Lens()
processor = LensProcessor()
with torch.no_grad():
    samples = processor([raw_image],[question])
    output = lens(samples)
    import pdb; pdb.set_trace()


