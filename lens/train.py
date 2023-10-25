import requests
from lens import Lens, LensProcessor
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "What is the image about?"

lens = Lens()
processor = LensProcessor()
with torch.no_grad():
    samples = processor([raw_image],[question])
    output = lens(samples)
print(output["prompts"])

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small",truncation_side = 'left',padding = True)
LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
outputs = LLM_model.generate(input_ids)
print(tokenizer.decode(outputs[0]))