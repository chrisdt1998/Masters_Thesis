## This file contains masking head technique for ViT based on the LTH but by using attn * grad_attn. This is approach
# differs because it uses an algorithm to decide what the next pruning percentage should be based on the validation loss.
import logging
import os
import argparse
import time

import numpy as np
from tqdm import tqdm
import torch

from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments, AutoConfig
from datasets import load_dataset, load_metric

logger = logging.getLogger(__name__)
# logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True
logging.getLogger().setLevel('CRITICAL')
logger.disabled = True

def pretrained_score(args, model, eval_dataloader):
    preds = None
    labels = None
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=True)):
        input_ids = batch['pixel_values'].to(args.device)
        label_ids = batch['labels'].to(args.device)

        outputs = model(pixel_values=input_ids, labels=label_ids, output_attentions=True)

        loss, logits, attentions = (outputs.loss, outputs.logits, outputs.attentions)  # Loss and logits

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

    return preds, labels


def compute_score(preds, labels):
    total_count = 0
    correct_count = 0
    for prediction, label in zip(preds, labels):
        predicted_label = np.argmax(prediction)
        if int(label) == predicted_label:
            correct_count += 1
        total_count += 1
    return correct_count / total_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='Name of dataset to be loaded from huggingface datasets. Default is cifar100.')
    # parser.add_argument('--output_dir', type=str, default='saved_numpys',
    #                     help='Output dir for the head masks')
    parser.add_argument('--dont_save_mask_all_iterations', action='store_true',
                        help='Call this to not save mask each iteration.')
    parser.add_argument('--initial_pruning_factor', type=int, default=0.2, help='Prune pruning_factor*remaining weights')
    parser.add_argument('--pruning_threshold', type=int, default=0.92,
                        help='Factor of the original validation accuracy that the next validation accuracy should not go below.')
    parser.add_argument('--dont_normalize_importance_by_layer', action='store_true')
    parser.add_argument('--dont_normalize_global_importance', action='store_true')
    parser.add_argument('--model_name', default='starting_checkpoint/checkpoint-100',
                        help='Model name to load when rolling back weights')
    parser.add_argument('--model_type', type=str, default='facebook/deit-base-patch16-224',
                        help='Choose from either facebook/deit-base-patch16-224 or google/vit-base-patch16-224-in21k')
    parser.add_argument('--iteration_id', type=str, default='base_run',
                        help='Iteration id to be able to run multiple scripts at the same time and identify.')
    args = parser.parse_args()
    args.output_dir = 'saved_numpys_' + args.iteration_id
    # args.num_epochs = int(100/args.initial_masking_factor) + 1
    args.num_epochs = 9
    if not os.path.isdir(args.output_dir):
        os.makedirs('saved_numpys_' + args.iteration_id)

    no_cuda = False
    local_rank = -1
    # Setup devices and distributed training
    if local_rank == -1 or no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Load pretrained model
    # Initializing a ViT vit-base-patch16-224 style configuration
    if args.dataset_name == 'mnist':
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_type, image_mean=0.5,
                                                                image_std=0.5)
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_type)

    train_ds, test_ds = load_dataset(args.dataset_name, split=['train', 'test'])
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        if args.dataset_name == 'mnist':
            inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
            inputs['pixel_values'] = torch.stack(
                [inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values']], dim=1)

            # Don't forget to include the labels!
            inputs['labels'] = example_batch['label']

        elif args.dataset_name == 'cifar100' or args.dataset_name == 'cifar10':
            inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

            if args.dataset_name == 'cifar10':
                inputs['labels'] = example_batch['label']
            else:
                inputs['labels'] = example_batch['fine_label']

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

    ## MNIST
    if args.dataset_name == 'mnist': labels = train_ds.features['label'].names

    ## CIFAR100
    if args.dataset_name == 'cifar100': labels = train_ds.features['fine_label'].names

    ## CIFAR10
    if args.dataset_name == 'cifar10': labels = train_ds.features['label'].names

    training_args = TrainingArguments(output_dir="./starting_checkpoint_" + args.dataset_name + '_' + args.iteration_id,
                                      per_device_train_batch_size=16, evaluation_strategy="no", max_steps=100,
                                      fp16=True, save_steps=100, eval_steps=100, logging_steps=100, learning_rate=2e-4,
                                      save_total_limit=2, remove_unused_columns=False, push_to_hub=False,
                                      report_to='tensorboard', load_best_model_at_end=False, disable_tqdm=True,
                                      log_level='critical', )

    model = ViTForImageClassification.from_pretrained(args.model_type, num_labels=len(labels),
                                                      id2label={str(i): c for i, c in enumerate(labels)},
                                                      label2id={c: str(i) for i, c in enumerate(labels)},
                                                      ignore_mismatched_sizes=True)


    args.collate_fn = collate_fn
    args.compute_metrics = compute_metrics
    args.transformed_train_ds = transformed_train_ds
    args.transformed_val_ds = transformed_val_ds
    args.feature_extractor = feature_extractor

    trainer = Trainer(model=model, args=training_args, data_collator=collate_fn, compute_metrics=compute_metrics,
                      train_dataset=transformed_train_ds, eval_dataset=transformed_val_ds,
                      tokenizer=feature_extractor, )

    processed_val_ds = trainer.get_eval_dataloader()

    preds, labels = pretrained_score(args, model, processed_val_ds)
    print(compute_score(preds, labels))




if __name__ == "__main__":
    main()
