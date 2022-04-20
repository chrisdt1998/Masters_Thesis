from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments, PretrainedConfig
from datasets import load_dataset, load_metric
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--head_mask', type=str,
                    help='File name of numpy array containing head mask in shape num_layers x num_heads')
parser.add_argument('--mask_heads', action="store_true",
                    help='Bool if to mask heads or not.')
args = parser.parse_args()

print(f"args.mask_heads: {args.mask_heads} \n args.head_mask: {args.head_mask}")


available_gpus = [torch.cuda.get_device_properties(torch.cuda.device(i)) for i in range(torch.cuda.device_count())]
device_ids = [i for i in range(torch.cuda.device_count())]
print('Available GPUs:')
for gpu in available_gpus:
        print(gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device =', device)
dataset_name = 'cifar100'
# dataset_name = 'mnist

if dataset_name == 'mnist':
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', image_mean=0.5,
                                                            image_std=0.5)
elif dataset_name == 'cifar100':
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')


train_ds, test_ds = load_dataset(dataset_name, split=['train', 'test'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']
print(train_ds.info)
print(train_ds.shape)

print(feature_extractor)


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    if dataset_name == 'mnist':
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        inputs['pixel_values'] = torch.stack([inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values']],
                                             dim=1)

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['label']

    elif dataset_name == 'cifar100':
        inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['fine_label']

    return inputs

processed_train_ds = train_ds.with_transform(transform)
processed_test_ds = test_ds.with_transform(transform)
processed_val_ds = val_ds.with_transform(transform)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def numpy_to_dict(head_mask):
    out_dict = {}
    for i, layer in enumerate(head_mask):
        heads_to_prune = []
        for j, head in enumerate(layer):
            if head == 0:
                heads_to_prune.append(j)
        out_dict[i] = heads_to_prune
    return out_dict


## MNIST
if dataset_name == 'mnist': labels = train_ds.features['label'].names

## CIFAR
if dataset_name == 'cifar100': labels = train_ds.features['fine_label'].names


model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)


head_mask = None
print(f"args.mask_heads: {args.mask_heads} \n args.head_mask: {args.head_mask}")
if args.mask_heads:
    head_mask = np.load(args.head_mask)
    print(f"Head masked {args.head_mask} loaded.")
    # head_mask = [[0, 0, 0, 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    #             [0, 0, 0, 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #             [0, 0, 0, 0., 0., 1., 0., 1., 0., 0., 0., 0.],
    #             [0, 0, 0, 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #             [0, 1, 0, 0., 1., 0., 0., 0., 0., 0., 0., 1.],
    #             [0, 0, 0, 0., 0., 0., 0., 1., 0., 1., 0., 0.],
    #             [0, 0, 0, 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #             [0, 0, 0, 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #             [0, 0, 0, 0., 0., 1., 1., 0., 0., 0., 0., 0.],
    #             [0, 0, 0, 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    #             [0, 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #             [0, 0, 0, 0., 0., 0., 0., 1., 1., 0., 0., 0.]]
    head_mask = np.array(head_mask)
    print(head_mask)
    head_mask_dict = numpy_to_dict(head_mask)
    print(head_mask_dict)

    model.prune_heads(head_mask_dict)

training_args = TrainingArguments(
  output_dir="./vit_full",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=1000,
  eval_steps=1000,
  logging_steps=100,
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