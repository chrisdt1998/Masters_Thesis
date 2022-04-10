#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the entropy of the head attentions
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import ViTModel, ViTConfig
from transformers.models.vit import modeling_vit

"""
from run_glue import ALL_MODELS, MODEL_CLASSES, load_and_cache_examples, set_seed
# from transformers import glue_compute_metrics as compute_metrics
from glue_metrics import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from experiment_impact_tracker.compute_tracker import ImpactTracker
from model_bert import BertForSequenceClassification
from config_bert import BertConfig
"""


logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True
output_dir = 'saved_numpys'
device = 'find gpu device here'
save_mask_all_iterations = True
masking_threshold = 0.1 ## Not sure what number to put here


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None
):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)

    preds = None
    labels = None
    tot_tokens = 0.0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask
        )
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

        tot_tokens += input_mask.float().detach().sum().data

    if compute_entropy:
        # Normalize
        attn_entropy /= tot_tokens
        np.save(os.path.join(output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
        logger.info("Attention entropies")
        print_2d_tensor(attn_entropy)
    if compute_importance:
        # Normalize
        head_importance /= tot_tokens
        # Layerwise importance normalization
        if not dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (
                        head_importance.max() - head_importance.min())

        # Print/save matrices
        np.save(os.path.join(output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


def mask_heads(model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(model, eval_dataloader, compute_entropy=False)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))
    original_score = 10 # TODO: test model on small test dataset
    current_score = original_score
    i = 0
    while current_score >= original_score * masking_threshold:
        head_mask = new_head_mask.clone()  # save current head mask
        if save_mask_all_iterations:
            np.save(os.path.join(output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.bert.config.num_attention_heads
            head_idx = head.item() % model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())

        if not selected_heads_to_mask:
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))

        # new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        logger.info(
            "Masking: remaining heads %d (%.1f percents)",
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask

def main():
    output_dir = 'path/to/output'
    device = 'put gpu device here, if gpu available'

    try_masking = True
    use_train_data = True
    masking_threshold = .1
    # Load pretrained model
    # Initializing a ViT vit-base-patch16-224 style configuration
    configuration = ViTConfig()

    # Initializing a model from the vit-base-patch16-224 style configuration
    model = ViTModel(configuration)


    ## TODO: Load weights of pretrained model.
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Distributed and parallel training
    model.to(device)

    # Print/save training arguments
    torch.save(os.path.join(output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s")

    # Prepare dataset
    if use_train_data:
        train_data =
        eval_data = random_split(train_data, [true_eval_len, len(train_data) - true_eval_len])[0]
    eval_sampler = SequentialSampler(eval_data) if local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    if try_masking and masking_threshold > 0.0 and masking_threshold < 1.0:
        head_mask = mask_heads(model, eval_dataloader)
    else:
        # Compute head entropy and importance score
        compute_heads_importance(model, eval_dataloader)

    # Head masks needs to be dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
    model._prune_heads(head_mask)

if __name__ == "__main__":
    main()
