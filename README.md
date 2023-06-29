# LENS🔍

[[Blog]](https://contextual.ai/introducing-lens) [[Demo]](https://lens.contextual.ai/) [[Paper]](https://arxiv.org/abs/2306.16410) [[Colab]](https://colab.research.google.com/github/ContextualAI/lens/blob/main/notebooks/example_usage.ipynb)

This is the official repository for the LENS (Large Language Models Enhanced to See) [paper](https://arxiv.org/abs/2306.16410). 

## Setup

1. We reccomend that you get a machine with GPUs and CUDA.
   A machine with a single GPU or even only a CPU works,
   although for large datasets you should get several GPUs.


2.  Create a python 3.9 conda or virtual environment and then install this repo as a pip package with:
    ```bash
    pip install llm-lens
    ```

## Usage

First, the system runs a series of highly descriptive vision modules on your own dataset of images. These modules provide a large set of natural language captions, tags, objects, and attributes for each image. These natural language descriptions are then provided to a large language model (LLM) so that it can solve a variety of tasks relating to the image. Despite the simplicity of our system, and the fact that it requires no finetuning, we demonstrate in our paper that it often performs better than other SOTA image-language models such as FLAMINGO, CLIP, and KOSMOS.


### Visual Descriptions

To simply get visual descriptions from LENS to pass on to an LLM, follow:

```python
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
    prompts = output["prompts"]
```

The generated prompts can be passed to the downstream LLM for the vision task that you want to solve. The `output` object also contains other useful information (tags, attributes, objects) that can be used to generate your custom prompts.

### Advanced Use Cases

+ Pass the image descriptions to a language model, and ask it a question

<!-- #region -->
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small",truncation_side = 'left',padding = True)
LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
outputs = LLM_model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```
<!-- #endregion -->

+ You can also augment your huggingface dataset

<!-- #region -->
```python
from datasets import load_dataset
from lens import Lens, LensProcessor

lens = Lens()
processor = LensProcessor()
ds = load_dataset("llm-lens/lens_sample_test", split="test")   
output_ds = lens.hf_dataset_transform(ds, processor, return_global_caption = False)
```

## Architecture

### LENS for open ended visual question answering
![lens_open_ended](https://github.com/ContextualAI/lens/assets/20826878/e2e9d993-3ae8-43d8-9152-0e73340afa41)


### LENS for image classification
![lens_close_ended](https://github.com/ContextualAI/lens/assets/20826878/45f3ae43-e3e4-48fc-b424-1899445d5c6f)


## Coming Soon

We are working on bringing the complete LENS experience to you and following scripts will be added soon:

- [ ] Evaluation on VQA and other datasets present in the paper.
- [ ] Generating vocabularies used in the paper.
- [ ] Other scripts needed to reproduce the paper.


## Citation

```
@misc{berrios2023language,
      title={Towards Language Models That Can See: Computer Vision Through the LENS of Natural Language}, 
      author={William Berrios and Gautam Mittal and Tristan Thrush and Douwe Kiela and Amanpreet Singh},
      year={2023},
      eprint={2306.16410},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
<!-- #endregion -->
