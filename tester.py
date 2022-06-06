# from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments, AutoConfig
# from datasets import load_dataset, load_metric
#
# import numpy as np
# import torch
#
# train_ds, test_ds = load_dataset('cifar100', split=['train', 'test'])
# splits = train_ds.train_test_split(test_size=0.1, shuffle=False)
# train_ds = splits['train']
# val_ds = splits['test']
#
# print(train_ds)
#
# feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')
#
# def transform(example_batch):
#     """
#     This method transformers the dataset applying the correct data augmentations to the dataset.
#     """
#
#     inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
#
#     inputs['labels'] = example_batch['label']
#
#     return inputs
#
#
# transformed_train_ds = train_ds.with_transform(transform)
# print(transformed_train_ds)
# transformed_test_ds = test_ds.with_transform(transform)
# transformed_val_ds = val_ds.with_transform(transform)
#
#
# def collate_fn(batch):
#     """
#     This method batches the images and labels.
#     """
#     return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#             'labels': torch.tensor([x['labels'] for x in batch])}
#
#
# metric = load_metric("accuracy")
#
#
# def compute_metrics(p):
#     """
#     This method is used by the trainer to compute the metrics when validating/testing.
#     """
#     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
#
#
# labels = train_ds.features['fine_label'].names
#
#
#
# print(f"Starting checkpoint does not exist. Creating one now.")
#
# # Initial model. Train for 100 steps.
# model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224', num_labels=len(labels),
#                                                   id2label={str(i): c for i, c in enumerate(labels)},
#                                                   label2id={c: str(i) for i, c in enumerate(labels)},
#                                                   ignore_mismatched_sizes=True)
#
# prune_test = {5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# model.prune_heads(prune_test)

import torch
import numpy as np
x = torch.randint(10, 11, (12, 12))
y = np.random.randint(2, size=(12, 12))
