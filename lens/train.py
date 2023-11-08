from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from utils import create_prompt_sample
import torch.nn.functional as F
from datasets import Dataset, load_dataset
import wandb

img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "What is the image about?"
gt_answer = "A pristine mountain range with a clear lake and sky"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lens_model = Lens()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

class LensTrainer():

    def compute_llm_likelihood(self, samples):
        question, tags = samples["questions"][0], samples["tags"][0]
        #Encode prompts and groundtruth answers
        prompts = [
            create_prompt_sample(samples, idx, mode="tag_only", question_prompt=question)
            for idx in range(len(tags))
        ]
        answers = [gt_answer for idx in range(len(tags))]
        prompt_encodings = tokenizer(prompts, return_tensors="pt", padding=True)
        answer_encodings = tokenizer(answers, return_tensors="pt", padding=True)
        #Get logits for groundtruth sequence when conditioned on each prompt
        outputs = llm_model(
            input_ids=prompt_encodings["input_ids"],
            attention_mask=prompt_encodings["attention_mask"],
            labels=answer_encodings["input_ids"]
        )
        #Compute likelihood based on logits
        logprobs = outputs.logits.log_softmax(dim=-1)
        masked_logprobs = logprobs.gather(dim=-1, index=answer_encodings["input_ids"].unsqueeze(-1))
        log_likelihood = masked_logprobs.squeeze().sum(dim=-1)
        return log_likelihood.softmax(dim=0)

    def compute_loss(self, model, inputs):
        samples = model(inputs)
        tags_likelihood = samples["top_scores"].squeeze().softmax(dim=0)
        llm_likelihood = self.compute_llm_likelihood(samples)
        kl_penalty = F.kl_div(
            torch.log(tags_likelihood), llm_likelihood, reduction="batchmean"
        )
        wandb.log({"kl_penalty": kl_penalty})
        return kl_penalty

def main():
    print(f"\nQuestion: {question} Groundtruth answer: {gt_answer}\n")
    wandb.init(project="lens-training-1-example")
    optimizer = torch.optim.Adam(lens_model.parameters(), lr=0.01)
    lensTrainer = LensTrainer()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(10):
        optimizer.zero_grad()
        inputs = processor([raw_image], [question])
        loss = lensTrainer.compute_loss(lens_model, inputs)
        wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
