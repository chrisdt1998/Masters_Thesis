from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch
import numpy as np


# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', image_mean=0.5, image_std=0.5)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')


train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']
print(train_ds.info)
print(train_ds.shape)

metric = load_metric("accuracy")

print(feature_extractor)

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')
    # inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
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

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
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
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(processed_test_ds)
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)