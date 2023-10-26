import os
from pathlib import Path
from typing import Any, List, Optional

import huggingface_hub
import open_clip
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    CLIPProcessor
)

from utils import (
    create_dataloader,
    create_prompt_sample,
    create_sampler,
    default_device,
)


def flatten(l):
    return [item for sublist in l for item in sublist]


class Lens(nn.Module):
    def __init__(
        self,
        clip_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        blip_name: str = "Salesforce/blip-image-captioning-large",
        attributes_weights: str = "zw_attributes_laion_ViT_H_14_2B_descriptors_text_davinci_003_full.pt",
        tags_weights: str = "zw_tags_laion_ViT_H_14_2B_vocab_lens.pt",
        vocab_attributes: str = "llm-lens/descriptors-text-davinci-003",
        vocab_tags: str = "llm-lens/vocab_tags",
        split_attributes: str = "full",
        split_tags: str = "train",
        load_8bit: bool = False,
        device: torch.device = default_device,
    ):
        super().__init__()
        # Load Base models
        self.device = device
        self.clip_name = clip_name
        self.blip_name = blip_name
        if self.clip_name is not None:
            self.clip_model = self.load_clip_model(self.clip_name, self.device)
            # Load weights
            huggingface_hub.hf_hub_download(
                repo_id="llm-lens/attributes",
                filename=attributes_weights,
                local_dir=str(Path(Path(__file__).resolve().parent) / "weights"),
            )
            huggingface_hub.hf_hub_download(
                repo_id="llm-lens/tags",
                filename=tags_weights,
                local_dir=str(Path(Path(__file__).resolve().parent) / "weights"),
            )

            self.attributes_weights = torch.load(
                str(
                    Path(Path(__file__).resolve().parent)
                    / f"weights/{attributes_weights}"
                ),
                map_location=self.device,
            ).float()
            self.tags_weights = torch.load(
                str(Path(Path(__file__).resolve().parent) / f"weights/{tags_weights}"),
                map_location=self.device,
            ).float()
            # Load Vocabularies
            self.vocab_tags = load_dataset(vocab_tags, split=split_tags)[
                "prompt_descriptions"
            ]
            self.vocab_attributes = flatten(
                load_dataset(vocab_attributes, split=split_attributes)[
                    "prompt_descriptions"
                ]
            )

        if self.blip_name is not None:
            self.blip_model = self.load_caption_model(
                self.blip_name, load_8bit, self.device
            )
            self.blip_processor = AutoProcessor.from_pretrained(self.blip_name)

    def load_caption_model(
        self, model_name: str, load_8bit: bool, device: torch.device
    ):
        if load_8bit:
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                device_map={"": device},
                load_in_8bit=True,
            )
        else:
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            )
        model = model.eval()
        model = model.to(device)

        return model

    def load_clip_model(self, model_name: str, device: torch.device):
        if "openai" in model_name:
            model = CLIPModel.from_pretrained(model_name).to(device)

        elif "laion" in model_name:
            model = open_clip.create_model_and_transforms(model_name)[0].to(device)
        return model

    def __call__(
        self,
        samples: dict,
        num_tags: int = 5,
        num_attributes: int = 5,
        contrastive_th: float = 0.2,
        num_beams: int = 5,  # For beam search
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
        return_tags: bool = True,
        return_attributes: bool = True,
        return_global_caption: bool = True,
        return_intensive_captions: bool = True,
        return_complete_prompt: bool = True,
    ):
        if return_tags:
            samples = self.forward_tags(
                samples, num_tags=num_tags, contrastive_th=contrastive_th
            )
        if return_attributes:
            samples = self.forward_attributes(
                samples, num_attributes=num_attributes, contrastive_th=contrastive_th
            )
        if return_global_caption:
            samples = self.forward_caption(
                samples,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
            )
        if return_intensive_captions:
            samples = self.forward_intensive_caption(
                samples,
                max_length=max_length,
                min_length=min_length,
                top_k=top_k,
                num_captions=num_captions,
            )

        if return_complete_prompt:
            samples = self.create_prompt_from_samples(samples)

        return samples

    def forward_tags(
        self, samples: dict, num_tags: int = 5, contrastive_th: float = 0.2
    ):
        # Get Image Features
        tags = []
        try:
            image_features = self.clip_model.encode_image(
                samples["clip_image"].to(self.device)
            )
        except:
            image_features = self.clip_model.get_image_features(
                pixel_values=samples["clip_image"]
            )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = image_features @ self.tags_weights
        top_scores, top_indexes = text_scores.float().cpu().topk(k=num_tags, dim=-1)
        for scores, indexes in zip(top_scores, top_indexes):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.vocab_tags[index] for index in filter_indexes]
            else:
                top_k_tags = []
            tags.append(top_k_tags)
        samples[f"tags"] = tags
        samples[f"tags_scores"] = top_scores
        return samples

    def forward_attributes(
        self, samples: dict, num_attributes: int = 5, contrastive_th: float = 0.2
    ):
        # Get Image Features
        attributes = []
        try:
            image_features = self.clip_model.encode_image(
                samples["clip_image"].to(self.device)
            )
        except:
            image_features = self.clip_model.get_image_features(
                pixel_values=samples["clip_image"]
            )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = image_features @ self.attributes_weights
        top_scores, top_indexes = (
            text_scores.float().cpu().topk(k=num_attributes, dim=-1)
        )
        for scores, indexes in zip(top_scores, top_indexes):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.vocab_attributes[index] for index in filter_indexes]
            else:
                top_k_tags = []
            attributes.append(top_k_tags)
        samples[f"attributes"] = attributes
        return samples

    def forward_caption(
        self,
        samples: dict,
        num_beams: int = 5,
        max_length: int = 30,
        min_length: int = 10,
    ):
        # Beam search
        captions_list = []
        pixel_values = samples["blip_image"].to(self.device, self.blip_model.dtype)
        input_ids = samples["blip_input_ids"].to(self.device)
        captions_ids = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            do_sample=False,
            num_beams=num_beams,
            top_p=1,
            max_length=max_length,
            min_length=min_length,
        )

        captions = self.blip_processor.batch_decode(
            captions_ids, skip_special_tokens=True
        )

        for caption in captions:
            captions_list.append(caption[12:].strip())

        samples["caption"] = captions_list
        return samples

    def forward_intensive_caption(
        self,
        samples: dict,
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
    ):
        pixel_values = samples["blip_image"].to(self.device, self.blip_model.dtype)
        input_ids = samples["blip_input_ids"].to(self.device)
        caption_ids = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=1,
            top_k=top_k,
            repetition_penalty=1,
            num_return_sequences=num_captions,
        )

        captions_text = self.blip_processor.batch_decode(
            caption_ids, skip_special_tokens=True
        )
        captions_text = [caption[12:].strip() for caption in captions_text]
        captions_text = [
            captions_text[i : i + num_captions]
            for i in range(0, len(captions_text), num_captions)
        ]
        samples["intensive_captions"] = captions_text
        return samples

    # This function could be more efficient
    def create_prompt_from_samples(
        self,
        samples: dict,
        mode: str = "all",  # vqa or vision or hm or or all
    ):
        num_samples = samples["clip_image"].shape[0]
        prompts = []
        for idx in range(num_samples):
            prompt = create_prompt_sample(samples, idx, mode=mode)

            prompts.append(prompt)
        samples["prompts"] = prompts
        return samples

    def hf_dataset_transform(
        self,
        ds: Dataset,
        processor: "LensProcessor",
        num_tags: int = 5,
        num_attributes: int = 5,
        contrastive_th: float = 0.2,
        num_beams: int = 5,  # For beam search
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
        return_tags: bool = True,
        return_attributes: bool = True,
        return_global_caption: bool = True,
        return_intensive_captions: bool = True,
        distributed_sampling: bool = False,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        dataset = LensDataset(ds, None, processor)
        # Create sampler
        sampler = create_sampler(dataset, distributed=distributed_sampling)
        # Create Dataloader
        dataloader = create_dataloader(
            dataset, sampler, batch_size=batch_size, num_workers=num_workers
        )

        # Get tags, attributes, caption, intensive_captions
        result = []
        for batch in dataloader:
            with torch.no_grad():
                batch = self(
                    batch,
                    num_tags=num_tags,
                    num_attributes=num_attributes,
                    contrastive_th=contrastive_th,
                    num_beams=num_beams,  # For beam search
                    max_length=max_length,
                    min_length=min_length,
                    top_k=top_k,
                    num_captions=num_captions,
                    return_tags=return_tags,
                    return_attributes=return_attributes,
                    return_global_caption=return_global_caption,
                    return_intensive_captions=return_intensive_captions,
                )

                keys = [
                    key
                    for key in batch.keys()
                    if key
                    in ["id", "tags", "attributes", "caption", "intensive_captions"]
                ]
                # print(f"keys: {keys}")
                for tuples in zip(*[batch[key] for key in keys]):
                    result.append(
                        {
                            k: (v.item() if k == "id" else v)
                            for k, v in zip(keys, tuples)
                        }
                    )

        if distributed_sampling is False:
            # To-Do: Add new columns to the huggingface dataset
            dict_ = {}
            for res in result:
                dict_[res["id"]] = {k: v for k, v in res.items() if k != "id"}

            # Map new columns would be faster
            def add_info(example):
                for component in [
                    "tags",
                    "attributes",
                    "caption",
                    "intensive_captions",
                ]:
                    try:
                        example[component] = dict_[example["id"]][component]
                    except:
                        pass
                return example

            result_ds = ds.map(add_info, batched=False)
            return result_ds
        else:
            # Only return the new componenets
            result_ds = Dataset.from_dict(
                {key: [d[key] for d in result] for key in result[0]}
            )
            return result_ds


class LensProcessor:
    def __init__(
        self,
        clip_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        blip_name: str = "Salesforce/blip-image-captioning-large",
    ):
        self.clip_processor = self.load_clip_transform(clip_name)
        self.blip_processor = AutoProcessor.from_pretrained(blip_name)

    def load_clip_transform(self, model_name: str):
        if "openai" in model_name:
            return CLIPProcessor.from_pretrained(model_name)

        elif "laion" in model_name:
            return open_clip.create_model_and_transforms(model_name)[2]

    def __call__(self, images: Any, questions: str):
        try:
            clip_image = torch.stack([self.clip_processor(image) for image in images])
        except:
            clip_image = self.clip_processor(images=images, return_tensors="pt")["pixel_values"]
        outputs = self.blip_processor(
            images=images, text=["a picture of"] * len(images), return_tensors="pt"
        )
        blip_image = outputs["pixel_values"]
        blip_input_ids = outputs["input_ids"]
        return {
            "clip_image": clip_image,
            "blip_image": blip_image,
            "blip_input_ids": blip_input_ids,
            "questions": questions,
        }


class LensDataset:
    def __init__(
        self,
        ds: Dataset,
        questions: Optional[List[str]] = None,
        processor: Optional[LensProcessor] = None,
    ):
        self.ds = ds
        self.processor = processor
        self.questions = questions

    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        id = self.ds[idx]["id"]
        try:
            question = self.ds[idx]["question"]
        except:
            pass
        try:
            question = self.questions[idx]
        except:
            question = ""
        outputs = self.processor([image], question)
        return {
            "id": torch.tensor(id, dtype=torch.int32),
            "clip_image": outputs["clip_image"].squeeze(0),
            "blip_image": outputs["blip_image"].squeeze(0),
            "blip_input_ids": outputs["blip_input_ids"].squeeze(0),
            "questions": outputs["questions"],
        }

    def __len__(self):
        return len(self.ds)
