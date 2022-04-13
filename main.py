from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments, PretrainedConfig
from datasets import load_dataset, load_metric
import torch
import numpy as np


available_gpus = [torch.cuda.get_device_properties(torch.cuda.device(i)) for i in range(torch.cuda.device_count())]
device_ids = [i for i in range(torch.cuda.device_count())]
print('Available GPUs:')
for gpu in available_gpus:
        print(gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device =', device)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', image_mean=0.5, image_std=0.5)
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
    inputs['pixel_values'] = torch.stack([inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values']], dim=1)
    # print(inputs['pixel_values'].shape)
    # inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    # print(inputs['pixel_values'].shape)

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
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

labels = train_ds.features['label'].names

load_head_mask = True
head_mask = None
if load_head_mask:
    head_mask = np.load(r"C:\Users\Gebruiker\Documents\GitHub\Masters_thesis\saved_numpys\head_importance_4.npy")
    print("Head masked loaded.")

# config = PretrainedConfig(
#     name_or_path='google/vit-base-patch16-224-in21k',
#     num_labels=len(labels),
#     id2label={str(i): c for i, c in enumerate(labels)},
#     label2id={c: str(i) for i, c in enumerate(labels)},
#     prune_heads=head_mask
# )
# model = ViTForImageClassification(config)
#
# print(config)
# exit(1)

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)


training_args = TrainingArguments(
  output_dir="./vit-full",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_train_ds,
    eval_dataset=processed_val_ds,
    tokenizer=feature_extractor,
    head_mask=head_mask
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(processed_test_ds)
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)