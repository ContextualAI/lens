'''
https://dienhoa.github.io/dhblog/posts/finetune_clip.html
https://huggingface.co/docs/transformers/training
'''
import argparse
import itertools
import math
import os
import pathlib
import random
import sys
import time

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pinecone

from dotenv import load_dotenv
from flax.training import checkpoints, train_state
from transformers import (
    FlaxAutoModel,
    FlaxAutoModelForCausalLM,
    AutoTokenizer,
)

from data import get_dataset, icl_openqa_prompt
from indexer import build_index

load_dotenv()
PINECONE_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENV)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_seed(seed=42, rank=0):
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Global batch size"
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained retriever with specified tag or file path.",
    )
    parser.add_argument(
        "--dataset",
        choices=["openwebtext", "wiki18", "squad"],
        default="openwebtext",
        help="Dataset to index."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--owt-bin", type=str, default=None, help="Path to OpenWebText bin file."
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Maximum length of document sequences."
    )
    parser.add_argument(
        "--train-docs", type=int, default=None, help="Maximum number of documents to train."
    )
    parser.add_argument(
        "--retrieval-docs", type=int, default=None, help="Maximum number of documents to index."
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train."
    )
    parser.add_argument(
        '--ckpt-dir', type=str, default=None, help='Path to save checkpoints.'
    )
    parser.add_argument(
        "--index-name", type=str, default=None, help="Index name to use."
    )

    args = parser.parse_args(args)
    return args


@jax.jit
def lm_likelihood(state, input_ids, attention_mask, logit_mask):
    """Log likelihood of the answer."""

    logits = state.apply_fn(
        input_ids,
        attention_mask,
        params=state.params,
    ).logits
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    masked_logprobs = jnp.where(logit_mask[..., None], logprobs, 0)
    likelihood = jnp.take_along_axis(masked_logprobs, input_ids[..., None], axis=2)
    likelihood = jnp.sum(likelihood[..., 0], axis=-1)
    return likelihood


@jax.jit
def average_pool(last_hidden_state, attention_mask):
    last_hidden = jnp.where(attention_mask[..., None] == 0, 0, last_hidden_state)
    return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]


@jax.jit
def encode_queries(state, input_ids, attention_mask):
    output = state.apply_fn(
        input_ids,
        attention_mask,
        params=state.params,
    ).last_hidden_state
    return average_pool(output, attention_mask)


@jax.jit
def train_step(
    state,
    query_embeddings,
    document_input_ids,
    document_attention_mask,
    lm_doc_scores,
    temperature,
):

    def loss_fn(params):
        # Encode passages
        passage_states = state.apply_fn(
            document_input_ids,
            document_attention_mask,
            params=params,
        ).last_hidden_state
        passage_embeddings = average_pool(passage_states, document_attention_mask)

        batch_size = query_embeddings.shape[0]
        n_docs = passage_embeddings.shape[0] // batch_size

        # Compute cosine scores
        norm_pe = passage_embeddings / \
            jnp.linalg.norm(passage_embeddings, axis=-1, keepdims=True)
        norm_pe = norm_pe.reshape(batch_size, n_docs, -1)
        norm_qe = query_embeddings / \
            jnp.linalg.norm(query_embeddings, axis=-1, keepdims=True)
        ret_doc_scores = jnp.sum(norm_qe[:, None, :] * norm_pe, axis=-1)

        # Compute KL divergence
        ret_dist = distrax.Softmax(jax.nn.log_softmax(ret_doc_scores), temperature)
        lm_dist = distrax.Softmax(lm_doc_scores, temperature)
        kl_div = ret_dist.kl_divergence(lm_dist)
        loss = jnp.mean(kl_div)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, loss


def encode_and_retrieve(*, state, tokenizer, index, batch, query_length, k):
    batch_q = tokenizer(
        [f'query: {q}' for q in batch['question']],  # TODO: making prefix configurable for E5
        padding='max_length',
        truncation=True,
        max_length=query_length,
        return_tensors='np',
    )
    query = encode_queries(
        state,
        batch_q.input_ids,
        batch_q.attention_mask)
    index_results = index.query(
        queries=jax.device_put(query).tolist(),
        top_k=k,
        include_metadata=True).results
    top_2k = np.array([[m.metadata['text'] for m in r.matches] for r in index_results])
    return query, top_2k


def create_learning_rate_fn(config):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.lr_warmup_steps,
    )
    num_batches = config.train_num_samples // config.batch_size
    train_steps = config.epochs * num_batches
    if config.lr_cosine_decay:
        decay_steps = train_steps - config.lr_warmup_steps
        opt_fn = optax.cosine_decay_schedule(
            init_value=config.learning_rate, decay_steps=decay_steps
        )
    else:
        opt_fn = optax.constant_schedule(config.learning_rate)

    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, opt_fn], boundaries=[config.lr_warmup_steps]
    )
    return learning_rate_fn


def main(args):
    args = parse_args(args)
    random_seed(args.seed, 0)
    world_size = jax.device_count()

    assert args.ckpt_dir is not None, 'Checkpoint directory must be specified.'
    ckpt_dir = pathlib.Path(os.path.expanduser(args.ckpt_dir))
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)

    # Load dataset
    train_docs = args.train_docs
    retrieval_docs = args.retrieval_docs
    doc_length = args.max_length
    dataset_name = args.dataset
    index_name = args.index_name if args.index_name is not None else dataset_name

    if train_docs is not None and train_docs % args.batch_size != 0:
        prev_train_docs = train_docs
        train_docs = (train_docs // args.batch_size) * args.batch_size
        print('Dropping last partial batch to avoid recompile '
              f'(prev: {prev_train_docs}, new: {train_docs})')

    # TODO: this is horrible logic lol
    if dataset_name == 'openwebtext':
        if retrieval_docs is not None:
            train_docs = args.train_docs \
                if dataset_name in ['openwebtext', 'squad'] else 0

        train_dataset, dataset = get_dataset(
            dataset_name,
            doc_length=doc_length * 2,
            split='train',
            train_docs=train_docs,
            retrieval_docs=retrieval_docs,
            owt_bin=args.owt_bin,
        )
        print('Train examples:', train_dataset.index_list)
        print('Retrieval examples:', dataset.index_list)
    elif dataset_name == 'squad':
        # NOTE: This will come pre-shuffled
        train_dataset, _ = get_dataset(
            dataset_name,
            doc_length=doc_length,
            split='train',
            train_docs=train_docs,
            retrieval_docs=0,
        )
        _, dataset = get_dataset(
            dataset_name,
            doc_length=doc_length,
            split='all',
            train_docs=0,
            retrieval_docs=args.retrieval_docs,
            dedup=dataset_name == 'squad',
        )
    else:
        raise ValueError(f'Supported dataset: {dataset_name}')

    print(f'Loaded {len(train_dataset):,} training documents from {dataset_name}')
    print(f'Loaded {len(dataset):,} retrieval documents from {dataset_name}')

    # Load ICL demonstrations
    icl_dataset, _ = get_dataset(
        dataset_name,
        doc_length=args.max_length,
        split='train',
        train_docs=3,  # 3-shot in-context learning
        retrieval_docs=0,
        owt_bin=args.owt_bin,
    )
    icl_questions, icl_contexts, icl_answers = [], [], []
    for i in range(len(icl_dataset)):
        icl_questions.append(icl_dataset[i]['question'])
        icl_contexts.append(icl_dataset[i]['text'])
        icl_answers.append(icl_dataset[i]['answers']['text'][0])
    icl_demos = list(zip(icl_questions, icl_contexts, icl_answers))

    # Load model parameters
    retriever = FlaxAutoModel.from_pretrained('intfloat/e5-small', from_pt=True)
    retriever.params = retriever.to_bf16(retriever.params)
    ret_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small', use_fast=True)

    gpt = FlaxAutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m')
    gpt.params = gpt.to_bf16(gpt.params)
    gpt_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m', use_fast=True)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_tokenizer.padding_side = 'left'

    ret_state = train_state.TrainState.create(
        apply_fn=retriever.__call__,
        params=retriever.params,
        tx=optax.adam(3e-5),
    )
    gpt_state = train_state.TrainState.create(
        apply_fn=gpt.__call__,
        params=gpt.params,
        tx=optax.adamw(0, b1=0.9, b2=0.95, weight_decay=0.01),
    )

    # Build initial index
    build_index(
        retriever,
        ret_tokenizer,
        dataset_name,
        dataset,
        doc_length=doc_length,
        batch_size=args.batch_size,
        index_name=index_name,
    )
    pc_index = pinecone.Index(index_name, pool_threads=world_size * 8)

    # Shard GPT
    p_lm_likelihood = jax.pmap(lm_likelihood, axis_name='batch')
    gpt_state = flax.jax_utils.replicate(gpt_state)

    # Shard retriever
    p_train_step = jax.pmap(
        train_step,
        axis_name='batch',
        static_broadcasted_argnums=(5,))
    ret_state = flax.jax_utils.replicate(ret_state)

    # RePlug LSR training loop
    total_num_batches = len(train_dataset) // args.batch_size
    sample_digits = math.ceil(math.log(len(train_dataset) + 1, 10))
    for epoch in range(args.epochs):
        batch_time_m = AverageMeter()
        end = time.time()
        num_samples = 0

        for i in range(total_num_batches):
            batch = train_dataset[i * args.batch_size : (i + 1) * args.batch_size]

            # Retrieve top-2k documents
            n_docs = 2 * args.top_k
            query_embeddings, top_2k = encode_and_retrieve(
                state=flax.jax_utils.unreplicate(ret_state),
                tokenizer=ret_tokenizer,
                index=pc_index,
                batch=batch,
                query_length=doc_length,
                k=n_docs,
            )

            num_samples += query_embeddings.shape[0]

            # Generate supervision
            gpt_batch_size = 2
            total_gpt_batches = query_embeddings.shape[0] // gpt_batch_size
            if query_embeddings.shape[0] % gpt_batch_size != 0:
                raise ValueError(f'Make batch size a factor of {gpt_batch_size}')

            lm_supervision = []
            for j in range(total_gpt_batches):
                gpt_batch = {
                    k: v[j * gpt_batch_size : (j + 1) * gpt_batch_size]
                    for k, v in batch.items()
                }
                gpt_top_2k = top_2k[j * gpt_batch_size : (j + 1) * gpt_batch_size]
                prompts_no_answer = np.array([
                    [icl_openqa_prompt(icl_demos, q, doc) for doc in gpt_top_2k[k]]
                    for k, q in enumerate(gpt_batch['question'])
                ])
                prompts_answer = np.array([
                    [p + a['text'][0] for p in prompts_no_answer[k]]
                    for k, a in enumerate(gpt_batch['answers'])
                ])

                # Get LM likelihood for supervision
                batch_p = gpt_tokenizer(
                    prompts_no_answer.reshape(-1).tolist(),
                    padding='max_length',
                    truncation=True,
                    max_length=gpt.config.max_position_embeddings,
                    return_tensors='np',
                )
                batch_pa = gpt_tokenizer(
                    prompts_answer.reshape(-1).tolist(),
                    padding='max_length',
                    truncation=True,
                    max_length=gpt.config.max_position_embeddings,
                    return_tensors='np',
                )
                p_ids = batch_p.input_ids
                pa_ids = batch_pa.input_ids
                pa_attn_mask = batch_pa.attention_mask
                # roll for next-token prediction
                logit_mask = jnp.logical_not(jnp.roll(p_ids == pa_ids, -1, axis=1))

                # Shard global batch across devices
                input_batch_size = p_ids.shape[0]
                batch_size_per_device, ragged = divmod(input_batch_size, world_size)
                if ragged:
                    raise ValueError(f'Make 2*bs*k divisible by {world_size}')
                shape_prefix = (world_size, batch_size_per_device)
                pa_ids = pa_ids.reshape(shape_prefix + pa_ids.shape[1:])
                pa_attn_mask = pa_attn_mask.reshape(shape_prefix + pa_attn_mask.shape[1:])
                logit_mask = logit_mask.reshape(shape_prefix + logit_mask.shape[1:])
                lm_log_lik = p_lm_likelihood(gpt_state, pa_ids, pa_attn_mask, logit_mask)
                lm_log_lik = lm_log_lik.reshape(input_batch_size // n_docs, n_docs)
                lm_supervision.append(lm_log_lik)

            lm_supervision = jnp.concatenate(lm_supervision, axis=0)

            # Update the retriever parameters
            batch_top_2k = ret_tokenizer(
                [f'passage: {p}' for p in top_2k.reshape(-1).tolist()],
                padding='max_length',
                truncation=True,
                max_length=doc_length,
                return_tensors='np',
            )
            top_2k_input_ids = batch_top_2k.input_ids
            top_2k_attn_mask = batch_top_2k.attention_mask

            # Shard global batch across devices
            input_batch_size = query_embeddings.shape[0]
            batch_size_per_device, ragged = divmod(input_batch_size, world_size)
            if ragged:
                raise ValueError(f'Batch size needs to be divisible by {world_size}')
            doc_batch_size = top_2k_input_ids.shape[0]
            docs_per_device, ragged = divmod(doc_batch_size, world_size)
            if ragged:
                raise ValueError(f'Make 2*bs*k divisible by {world_size}')
            shape_prefix = (world_size, batch_size_per_device)
            doc_shape_prefix = (world_size, docs_per_device)
            query_embeddings = query_embeddings.reshape(shape_prefix + query_embeddings.shape[1:])
            top_2k_input_ids = top_2k_input_ids.reshape(doc_shape_prefix + top_2k_input_ids.shape[1:])
            top_2k_attn_mask = top_2k_attn_mask.reshape(doc_shape_prefix + top_2k_attn_mask.shape[1:])
            lm_supervision = lm_supervision.reshape(shape_prefix + lm_supervision.shape[1:])

            ret_state, lsr_loss = p_train_step(
                ret_state,
                query_embeddings,
                top_2k_input_ids,
                top_2k_attn_mask,
                lm_supervision,
                1,
            )
            lsr_loss = jnp.mean(lsr_loss)

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logging
            samples_per_second = args.batch_size * world_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            percent_complete = num_samples / len(train_dataset) * 100
            lr = jnp.array(0) #jax.tree_map(lambda x: x[0], learning_rate_fn(state.step))
            print(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{len(train_dataset)} ({percent_complete:.0f}%)] "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/xpu "
                f"LR: {lr.item():5f} Loss: {lsr_loss.item():.4f}"
            )

        epoch_params = flax.jax_utils.unreplicate(ret_state).params
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            step=epoch,
            target=epoch_params,
            overwrite=True,
            keep=args.epochs,
        )

        # Build index
        retriever.params = epoch_params
        build_index(
            retriever,
            ret_tokenizer,
            dataset_name,
            dataset,
            doc_length=doc_length,
            batch_size=args.batch_size,
            index_name=index_name,
        )


if __name__ == "__main__":
    main(sys.argv[1:])






# from model import Lens, LensDataset, LensProcessor
# import requests
# from lens import Lens, LensProcessor
# from PIL import Image
# from scipy.special import rel_entr
# from transformers import Trainer
# import torch
# import clip
# import numpy as np

# img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
# question = "What is the image about?"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class LensTrainer():
#     def __init__(self):
#         self.model = Lens()
#         self.processor = LensProcessor()
#         with torch.no_grad():
#             inputs = self.processor([raw_image],[question])
#         self.compute_loss(self.model, inputs)

#     def compute_loss(model, inputs):
#         total_loss = 0
#         lm_likelihood = torch.ones((5, 1))
#         for desc in ["tags", "attributes"]:
#             image_features = model.clip_model.encode_image(
#                 inputs["clip_image"].to(device)
#             )
#             text_features = model.clip_model.encode_text(
#                 clip.tokenize(inputs[desc][0]).to(device)
#             )
#             logits = text_features @ image_features.T 
#             desc_likelihood = logits.softmax(dim=0).to(device).numpy()
#             total_loss += rel_entr(lm_likelihood, desc_likelihood)

# def main():
#     trainer = LensTrainer()



