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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

bertscore = load("bertscore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def bert_coco_trained(coco_ex, descs=["tags", "attributes"]):
    wandb.init(project="lens-training-coco-dataset")
    true_captions = []
    output_captions = []

    for i in range(500):
        curr_ex = coco_ex[i]
        img_url = curr_ex['url']
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        question = "What is the image about?"

        samples = processor([raw_image],[question])
        output = lens_model(
            samples,
            num_tags=10,
            return_tags=("tags" in descs),
            num_attributes=10,
            return_attributes=("attributes" in descs),
        )
        print(output["prompts"])
        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        outputs = llm_model.generate(input_ids)
        llm_answer = tokenizer.decode(outputs[0])
        print(llm_answer)
        output_captions.append(llm_answer)
        
        true_captions.append(curr_ex['caption'])
        print(i, output_captions[i], true_captions[i])

    print(len(output_captions), len(true_captions))
    scores = bertscore.compute(predictions=output_captions, references=true_captions, lang="en")
    print(scores)
    return scores

def bert_vqa_trained(vqa_ex, descs=["tags", "attributes"]):
    wandb.init(project="lens-training-coco-dataset")
    true_answers = []
    output_captions = []

    print("DATASET LOADED, ABOUT TO ITERATE")

    for i in range(500):
        try:
            curr_ex = vqa_ex[i]
            img_url = curr_ex['flickr_original_url']
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            question = curr_ex['question']

            print({i})

            samples = processor([raw_image],[question])
            output = lens_model(
                samples,
                num_tags=10,
                return_tags=("tags" in descs),
                num_attributes=10,
                return_attributes=("attributes" in descs),
            )
            print(output["prompts"])
            input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
            outputs = llm_model.generate(input_ids)
            llm_answer = tokenizer.decode(outputs[0])
            print(llm_answer)
            output_captions.append(llm_answer)
            
            true_answers.append(curr_ex['answers'][0])
            print(i, output_captions[i], true_answers[i])
        except:
            print("Error, skipped example")

    print(len(output_captions), len(true_answers))
    scores = bertscore.compute(predictions=output_captions, references=true_answers, lang="en")
    print(scores)
    return scores

def get_examples(ds):
    examples = []
    for example in ds:
        examples.append(example)
        if len(examples) >= 500:
            break
    return examples

def main(descs=["tags", "attributes"]):
    coco_ds = load_dataset("RIW/small-coco", split="validation", streaming=True)
    coco_ex = get_examples(coco_ds)
    vqa_ds = load_dataset("textvqa", split="validation", streaming=True)
    vqa_ex = get_examples(vqa_ds)
    
    bert = {}
    bert['coco'] = bert_coco_trained(coco_ex, descs)
    print(f"{bert['coco']=}")
    bert['vqa'] = bert_vqa_trained(vqa_ex, descs)
    print(f"{bert['vqa']=}")

    average_precision = sum(bert['coco']['precision']) / len(bert['coco']['precision'])
    average_recall = sum(bert['coco']['recall']) / len(bert['coco']['recall'])
    average_f1 = sum(bert['coco']['f1']) / len(bert['coco']['f1'])

    print(f"Average Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")
    print(f"Average F1 Score: {average_f1}")

    average_precision2 = sum(bert['vqa']['precision']) / len(bert['vqa']['precision'])
    average_recall2 = sum(bert['vqa']['recall']) / len(bert['vqa']['recall'])
    average_f12 = sum(bert['vqa']['f1']) / len(bert['vqa']['f1'])

    print(f"Average Precision: {average_precision2}")
    print(f"Average Recall: {average_recall2}")
    print(f"Average F1 Score: {average_f12}")

    # Qualitative Analysis: top-k and bottom-k F1 scores
    k = 20
    sorted_indices = sorted(range(len(bert['coco']['f1'])), key=lambda i: bert['coco']['f1'][i], reverse=True)
    top_k_indices = sorted_indices[:k]
    bottom_k_indices = sorted_indices[-k:]
    bottom_k_indices = sorted(bottom_k_indices, key=lambda i: bert['coco']['f1'][i])

    print(f"Top {k} F1 Score Examples: {top_k_indices}")
    print(f"Bottom {k} F1 Score Examples: {bottom_k_indices}")

    sorted_indices = sorted(range(len(bert['vqa']['f1'])), key=lambda i: bert['vqa']['f1'][i], reverse=True)
    top_k_indices = sorted_indices[:k]
    bottom_k_indices = sorted_indices[-k:]
    bottom_k_indices = sorted(bottom_k_indices, key=lambda i: bert['vqa']['f1'][i])


if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--descriptions',
                        nargs='+',
                        help='Which descriptions to train on')
    args = parser.parse_args()
    descs = args.descriptions if args.descriptions else ["tags", "attributes"]
    model_path = "trained_model_" + "_".join(descs) + ".pt"
    lens_model.load_state_dict(torch.load(model_path))
    main(descs)

    