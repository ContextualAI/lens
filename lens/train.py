from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from utils import create_prompt_sample, create_dataloader, create_sampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def compute_llm_likelihood(samples, labels):
    tags = samples["tags"][0]
    #Encode prompts and groundtruth answers
    all_prompts = [
        create_prompt_sample(
            samples, idx, mode="one_tag_only", 
            question_prompt=samples["questions"][idx]
        ) for idx in range(len(tags))
    ]
    all_labels = [ labels for idx in range(len(tags)) ]
    prompt_encodings = tokenizer(all_prompts, return_tensors="pt", padding=True)
    label_encodings = tokenizer(all_labels, return_tensors="pt", padding=True)
    #Get logits for groundtruth sequence when conditioned on each prompt
    outputs = llm_model(
        input_ids=prompt_encodings["input_ids"],
        attention_mask=prompt_encodings["attention_mask"],
        labels=label_encodings["input_ids"]
    )
    #Compute likelihood based on logits
    logprobs = outputs.logits.log_softmax(dim=-1)
    masked_logprobs = logprobs.gather(dim=-1, index=label_encodings["input_ids"].unsqueeze(-1))
    log_likelihood = masked_logprobs.squeeze().sum(dim=-1)
    return log_likelihood.softmax(dim=0)

def compute_loss(samples, labels):
    tags_likelihood = samples["top_scores"].squeeze().softmax(dim=0)
    llm_likelihood = compute_llm_likelihood(samples, labels)
    kl_penalty = F.kl_div(
        torch.log(tags_likelihood), llm_likelihood, reduction="sum"
    )
    wandb.log({"kl_penalty": kl_penalty})
    return kl_penalty

def main():
    wandb.init(project="lens-training-coco-dataset")
    question = "What is the image about?"
    ds = load_dataset("RIW/small-coco", split="train")
    sampler = create_sampler(ds, distributed=False)
    dataloader = create_dataloader(ds, sampler, batch_size=1)
    optimizer = torch.optim.Adam(lens_model.parameters(), lr=1e-5)
    torch.autograd.set_detect_anomaly(True)
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = processor([batch['image']], [question])
            samples = lens_model(inputs)
            loss = compute_loss(samples, batch['caption'])
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    main()
