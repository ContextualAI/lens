from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from datasets import Dataset, load_dataset
import wandb
import re
from evaluate import load

bertscore = load("bertscore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


def compute_bert_pretrained(ds):
    wandb.init(project="lens-training-coco-dataset")
    true_captions = ds['caption'][:1]
    output_captions = []

    for i in range(1):
        curr_ex = next(iter(ds))
        img_url = curr_ex['url']
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        question = "What is the image about?"

        samples = processor([raw_image],[question])
        output = lens_model(samples)
        print(output["prompts"])
        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        outputs = llm_model.generate(input_ids)
        llm_answer = tokenizer.decode(outputs[0])
        print(llm_answer)
        output_captions.append(llm_answer)

    scores = bertscore.compute(predictions=output_captions, references=true_captions, lang="en")
    print(scores)
    return scores

def main():
    ds = [load_dataset("RIW/small-coco", split="validation", streaming=True)]
    # load_dataset("conceptual_captions", split="train", streaming=True),
    # load_dataset("zzliang/GRIT", split="train", streaming=True)]
    bert_pretrained = []
    bert_trained = []

    for i in range(1):
        bert_pretrained[i] = compute_bert_pretrained(ds[i])
       # bert_trained[i] = compute_bert_trained(ds[i])

    print(bert_pretrained)

if __name__ == "__main__":
    main()

    
