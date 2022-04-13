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
import logging
import os

import numpy as np
from tqdm import tqdm
import torch

from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric


logger = logging.getLogger(__name__)
# logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True
logging.getLogger().setLevel(logging.INFO)
output_dir = 'saved_numpys'
save_mask_all_iterations = True
masking_threshold = 0.9 ## keep masking until metric < threshold * original metric value
masking_amount = 0.1 ## amount of heads to mask at each masking step.
dont_normalize_importance_by_layer = True
dont_normalize_global_importance = True

no_cuda = False
local_rank = -1
# Setup devices and distributed training
if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = 0 if no_cuda else torch.cuda.device_count()
else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend


def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(model, eval_dataloader, compute_importance=True, head_mask=None
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
        input_ids = batch['pixel_values'].to(device)
        label_ids = batch['labels'].to(device)


        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            pixel_values=input_ids, head_mask=head_mask, labels=label_ids
        )

        # print(outputs)
        loss, logits = (
            outputs.loss,
            outputs.logits,
        )  # Loss and logits
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_importance:
            # print("Head importance, ", head_mask.grad.abs().detach())
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

        # tot_tokens += input_mask.float().detach().sum().data

    if compute_importance:
        # Normalize
        # TODO: Check how to normalize
        # head_importance /= tot_tokens

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
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
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
    _, head_importance, preds, labels = compute_heads_importance(model, eval_dataloader)

    print("Preds and labels shape", preds.shape, labels.shape)
    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))
    original_score = compute_score(preds, labels)
    current_score = original_score
    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * masking_threshold)
    i = 0
    while current_score >= original_score * masking_threshold:
        print(f"Original score {original_score}")
        print(f"Threshold: {original_score * masking_threshold}")
        head_mask = new_head_mask.clone()  # save current head mask
        if save_mask_all_iterations:
            np.save(os.path.join(output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            print("Break 1", len(current_heads_to_mask), num_to_mask)
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                print("Break 2", len(selected_heads_to_mask), num_to_mask)
                break
            layer_idx = head.item() // model.config.num_hidden_layers
            head_idx = head.item() % model.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())
            print(f"Layer {layer_idx}, head number {head_idx} is pruned. Number of heads pruned is: {len(selected_heads_to_mask)}")

        if not selected_heads_to_mask:
            print("Break 3", selected_heads_to_mask)
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))

        # new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            model, eval_dataloader, head_mask=new_head_mask
        )
        current_score = compute_score(preds, labels)
        logger.info("Masking: current score: %f, remaning heads %d (%.1f percents)", current_score, new_head_mask.sum(),
                    new_head_mask.sum() / new_head_mask.numel() * 100, )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask

def compute_score(preds, labels):
    total_count = 0
    correct_count = 0
    for prediction, label in zip(preds, labels):
        predicted_label = np.argmax(prediction)
        if int(label) == predicted_label:
            correct_count += 1
        total_count += 1
    return correct_count/total_count


def main():
    try_masking = True
    # Load pretrained model
    # Initializing a ViT vit-base-patch16-224 style configuration
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', image_mean=0.5,
                                                            image_std=0.5)
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    train_ds, test_ds = load_dataset('mnist', split=['train', 'test'])
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    print(train_ds.info)
    print(train_ds.shape)

    print(feature_extractor)

    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        # print(inputs['pixel_values'].shape)
        inputs['pixel_values'] = torch.stack([inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values']],
                                             dim=1)
        # print(inputs['pixel_values'].shape)
        # inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        # print(inputs['pixel_values'].shape)

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['label']
        return inputs

    transformed_train_ds = train_ds.with_transform(transform)
    transformed_test_ds = test_ds.with_transform(transform)
    transformed_val_ds = val_ds.with_transform(transform)

    def collate_fn(batch):
        return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])}

    metric = load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    labels = train_ds.features['label'].names

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)}, label2id={c: str(i) for i, c in enumerate(labels)})

    training_args = TrainingArguments(output_dir="./vit-full", per_device_train_batch_size=16,
        evaluation_strategy="steps", num_train_epochs=4, fp16=True, save_steps=100, eval_steps=100, logging_steps=10,
        learning_rate=2e-4, save_total_limit=2, remove_unused_columns=False, push_to_hub=False, report_to='tensorboard',
        load_best_model_at_end=True, )

    trainer = Trainer(model=model, args=training_args, data_collator=collate_fn, compute_metrics=compute_metrics,
        train_dataset=transformed_train_ds, eval_dataset=transformed_val_ds, tokenizer=feature_extractor, )

    processed_val_ds = trainer.get_eval_dataloader()

    # Distributed and parallel training
    model.to(device)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    if try_masking and masking_threshold > 0.0 and masking_threshold < 1.0:
        head_mask = mask_heads(model, processed_val_ds)
    else:
        # Compute head entropy and importance score
        compute_heads_importance(model, processed_val_ds)

    print(head_mask)

    # Head masks needs to be dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
    # When we finally have the mask, we can prune the heads and test performance speed gain.
    # model._prune_heads(head_mask)

if __name__ == "__main__":
    main()
