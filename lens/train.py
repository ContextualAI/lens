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
import matplotlib.pyplot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def compute_llm_likelihood(samples, labels, desc):
    batch_size, num_descs = np.array(samples[desc]).shape
    #Encode prompts and groundtruth answers
    all_prompts, all_labels = [], []
    for i in range(batch_size):
        for j in range(num_descs):
            all_prompts.append(create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_single",
                question_prompt=samples["questions"][i]
            ))
            all_labels.append(labels[i])
    prompt_encodings = tokenizer(all_prompts, return_tensors="pt", padding=True)
    label_encodings = tokenizer(all_labels, return_tensors="pt", padding=True)
    #Get logits for groundtruth sequence when conditioned on each prompt
    outputs = llm_model(
        input_ids=prompt_encodings["input_ids"],
        attention_mask=prompt_encodings["attention_mask"],
        labels=label_encodings["input_ids"]
    )
    #Compute likelihood based on logits
    _, seq_length, vocab_size = outputs.logits.shape
    logits = outputs.logits.reshape((batch_size, num_descs, seq_length, vocab_size))
    logprobs = logits.log_softmax(dim=-1)
    label_input_ids = label_encodings["input_ids"].reshape((batch_size, num_descs, seq_length, 1))
    masked_logprobs = logprobs.gather(dim=-1, index=label_input_ids)
    log_likelihood = masked_logprobs.squeeze().sum(dim=-1)
    return torch.exp(log_likelihood)

def compute_loss(samples, labels):
    loss = 0
    for desc in ["intensive_captions" "tags", "attributes"]:
        desc_likelihood = samples[f"top_scores_{desc}"].squeeze()
        llm_likelihood = compute_llm_likelihood(samples, labels, desc)
        kl_penalty = F.kl_div(
            desc_likelihood.log_softmax(dim=-1), llm_likelihood.log_softmax(dim=-1),
            reduction="batchmean", log_target=True
        )
        loss += kl_penalty
        wandb.log({f"kl_penalty_{desc}": kl_penalty})
    #plt.scatter(tags_likelihood, llm_likelihood)
    return loss

def train(num_epochs=100, lr=1e-5, batch_size=8, training_size=8):
    wandb.init(project="lens-training-coco-dataset")
    question = ["What is the image about" for i in range(batch_size)]
    ds = load_dataset("RIW/small-coco", split="train", streaming=True)
    dataloader = create_dataloader(ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(lens_model.parameters(), lr=lr)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            if i > (training_size // batch_size):
                continue
            optimizer.zero_grad()
            inputs = processor(batch['image'], 'question')
            samples = lens_model(inputs)
            loss = compute_loss(samples, batch['caption'])
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
