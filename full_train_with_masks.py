"""
This file contains the code for training multiple models using the mask loaded. The intention of the file was for the
experiments.

This file was created by and designed by Christopher du Toit.
"""

import torch
import numpy as np
import argparse

import os
import sys
sys.path.append(r"C:\Users\Gebruiker\Documents\GitHub\transformers\src")
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments, PretrainedConfig
from datasets import load_dataset, load_metric


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='cifar100',
                    help='Name of dataset to be loaded from huggingface datasets. Default is cifar100.')
parser.add_argument('--model_name', default='starting_checkpoint/checkpoint-100',
                    help='Model name to load when rolling back weights')
parser.add_argument('--iteration_id', default=[], nargs='+',
                    help='Iteration id list.')
parser.add_argument('--experiment_id', type=str, default='experiment_1',
                    help='Experiment id for the experiment we are currently running.')
args = parser.parse_args()

args.model_name = 'facebook/deit-base-patch16-224'

available_gpus = [torch.cuda.get_device_properties(torch.cuda.device(i)) for i in range(torch.cuda.device_count())]
device_ids = [i for i in range(torch.cuda.device_count())]
print('Available GPUs:')
for gpu in available_gpus:
        print(gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device =', device)


if args.dataset_name == 'mnist':
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name, image_mean=0.5,
                                                            image_std=0.5)
elif args.dataset_name == 'cifar100' or args.dataset_name == 'cifar10':
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name)


train_ds, test_ds = load_dataset(args.dataset_name, split=['train', 'test'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']


def transform(example_batch):
    """
    This method transformers the dataset applying the correct data augmentations to the dataset.
    """
    # Take a list of PIL images and turn them to pixel values
    if args.dataset_name == 'mnist':
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        inputs['pixel_values'] = torch.stack([inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values']],
            dim=1)

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['label']

    elif args.dataset_name == 'cifar100' or args.dataset_name == 'cifar10':
        inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

        if args.dataset_name == 'cifar10':
            inputs['labels'] = example_batch['label']
        else:
            inputs['labels'] = example_batch['fine_label']

    return inputs

processed_train_ds = train_ds.with_transform(transform)
processed_test_ds = test_ds.with_transform(transform)
processed_val_ds = val_ds.with_transform(transform)


def collate_fn(batch):
    """
    This method batches the images and labels.
    """
    return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])}

metric = load_metric("accuracy")

def compute_metrics(p):
    """
    This method is used by the trainer to compute the metrics when validating/testing.
    """
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def numpy_to_dict(head_mask):
    """
    This method converts the head_mask tensor into a dictionary which is the required format for pruning.
    :param head_mask: Head mask tensor.
    :type head_mask: tensor
    :return: Dictionary of head mask.
    :rtype: dict
    """
    out_dict = {}
    for i, layer in enumerate(head_mask):
        heads_to_prune = []
        for j, head in enumerate(layer):
            if head == 0:
                heads_to_prune.append(j)
        out_dict[i] = heads_to_prune
    return out_dict


## MNIST
if args.dataset_name == 'mnist': labels = train_ds.features['label'].names

## CIFAR
if args.dataset_name == 'cifar100': labels = train_ds.features['fine_label'].names

## CIFAR10
if args.dataset_name == 'cifar10': labels = train_ds.features['label'].names


for iteration_id in args.iteration_id:
    model = ViTForImageClassification.from_pretrained(args.model_name)

    if iteration_id != 'None':
        file_name = os.path.join(args.experiment_id, 'saved_numpys_' + iteration_id, 'head_mask.npy')
        head_mask = np.load(file_name)
        num_masked = 144 - np.sum(head_mask)
        print(f"Head mask for experiment, {args.experiment_id} and iteration {iteration_id} loaded. Number of heads "
              f"pruned is {num_masked}")
        head_mask = np.array(head_mask)
        print(head_mask)
        head_mask_dict = numpy_to_dict(head_mask)

        model.prune_heads(head_mask_dict)
    else:
        print(f"No head mask for experiment, {args.experiment_id}.")


    training_args = TrainingArguments(
      output_dir="./" + args.experiment_id + "_vit_mask_full/",
      per_device_train_batch_size=16,
      evaluation_strategy="steps",
      num_train_epochs=4,
      fp16=True,
      save_steps=5000,
      eval_steps=5000,
      logging_steps=500,
      learning_rate=2e-4,
      save_total_limit=2,
      remove_unused_columns=False,
      push_to_hub=False,
      report_to='tensorboard',
      load_best_model_at_end=True,
      disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=processed_train_ds,
        eval_dataset=processed_val_ds,
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(processed_test_ds)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    model.cpu()
