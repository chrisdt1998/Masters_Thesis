## This file contains masking head technique for ViT based on the LTH but without the head mask gradient.
import logging
import os
import argparse

import numpy as np
from tqdm import tqdm
import torch


from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric


logger = logging.getLogger(__name__)
# logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True
logging.getLogger().setLevel(logging.INFO)



def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(args, model, eval_dataloader, head_mask=None):
    # Prepare our tensors
    n_heads = 12
    n_layers = 12
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
    else:
        head_mask = head_mask.clone().detach()

    preds = None
    labels = None
    tot_tokens = 14 * 14
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=True)):
        input_ids = batch['pixel_values'].to(args.device)
        label_ids = batch['labels'].to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        # outputs = model(
        #     pixel_values=input_ids, head_mask=head_mask, labels=label_ids
        # )
        outputs = model(
            pixel_values=input_ids, labels=label_ids, output_attentions=True
        )

        loss, logits, attentions = (
            outputs.loss,
            outputs.logits,
            outputs.attentions
        )  # Loss and logits
        for attention_map in attentions:
            attention_map.retain_grad()
        loss.backward()
        attention_grads = [attention_map.grad.detach().cpu() for attention_map in attentions]
        attentions = [attention_map.detach().cpu() for attention_map in attentions]

        # Here we take the matrix multiplication between the attention head and the attention head gradient w.r.t.
        # the loss function as described in theory.
        # We take the shape [0, 1:] as it is the attention referring to the [CLS] token then take the sum of all
        # the inputs in that attention matrix to get our head sensitivity.
        all_heads_sensitivity = []
        for layer, layer_grad in zip(attentions, attention_grads):
            head_sensitivity = torch.zeros(layer.shape[1])
            for batch, batch_grad in zip(layer, layer_grad):
                for idx, (head, head_grad) in enumerate(zip(batch, batch_grad)):
                    head_sensitivity[idx] += torch.matmul(head, head_grad).abs()[0, 1:].sum(dim=0)
            all_heads_sensitivity.append(head_sensitivity)


        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

        # add attention heads to head_importance by simply taking the sum. The different indexes are to account for the
        # pruning of the heads.
        j_idx = 0
        row_visited = False
        for layer_idx in range(n_layers):
            i_idx = 0
            for head_idx in range(n_heads):
                if head_mask[layer_idx, head_idx] == 1:
                    row_visited = True
                    head_importance[layer_idx][head_idx] += all_heads_sensitivity[j_idx][i_idx]
                    i_idx += 1
            if row_visited:
                j_idx += 1
                row_visited = False

    # Normalize`
    # TODO: Check how to normalize
    # print(f"Head_Importance 1 {head_importance}")
    head_importance /= tot_tokens
    # print(f"Head_Importance 2 {head_importance}")


    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        # print(f"Head_Importance 3 {head_importance}")
    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (
                    head_importance.max() - head_importance.min())
        # print(f"Head_Importance 4 {head_importance}")

    # Print/save matrices
    np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

    logger.info("Head importance scores")
    print_2d_tensor(head_importance)
    logger.info("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=args.device
    )
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)

    return head_importance, preds, labels


def mask_heads(args):
    # Initialize the model and trainer
    model, trainer, eval_dataloader = reset_model(args)
    # Train the model for 1000 steps (about 33% of the training data).
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    # load trained model
    model = trainer.model
    head_importance, preds, labels = compute_heads_importance(model, eval_dataloader)

    print(f"Starting score {compute_score(preds, labels)}")
    new_head_mask = torch.ones_like(head_importance)
    i = 0
    final_scores = []
    output_heatmap_mask = np.zeros_like(new_head_mask)
    while i < args.num_epochs:
        head_mask = new_head_mask.clone()  # save current head mask
        if not args.dont_save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
            final_scores.append(compute_score(preds, labels))
            print(f"Iteration {i} score: {compute_score(preds, labels)}")
            print(f"Iteration {i} head mask: \n {head_mask}")
            print(f"Iteration {i} head importance: \n {head_importance}")
        i += 1
        # heads from least important to most - keep only not-masked heads
        unpruned_heads = torch.sum(new_head_mask)
        num_to_mask = max(1, int(unpruned_heads * args.masking_factor))
        print(f"num_to_mask for iter {i} is {num_to_mask}, total pruned already is {144 - unpruned_heads}, ({(144 - unpruned_heads)*100/144}%)")

        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            print("Break 1", len(current_heads_to_mask), num_to_mask)
            break

        print(f"Number of heads left: {torch.sum(new_head_mask)}")
        if torch.sum(new_head_mask) < 10:
            print(f"Only {torch.sum(new_head_mask)} number of heads left causing the algorithm to stop searching.")
            break

        # mask heads
        # print(f"Current heads to mask {current_heads_to_mask}")
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            print(head_importance.view(-1)[head.item()])

            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                print("Break 2", len(selected_heads_to_mask), num_to_mask)
                break
            layer_idx = head.item() // model.config.num_hidden_layers
            head_idx = head.item() % model.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            output_heatmap_mask[layer_idx][head_idx] = i
            selected_heads_to_mask.append(head.item())
            print(f"Layer {layer_idx}, head number {head_idx} is pruned. Number of heads pruned is: {len(selected_heads_to_mask)}")

        if not selected_heads_to_mask:
            print("Break 3", selected_heads_to_mask)
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))

        # new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Reinitialize the model and trainer
        model.cpu()
        model, trainer, eval_dataloader = reset_model(args)
        # Train the model for 1000 steps (about 33% of the training data).
        head_mask_dict = numpy_to_dict(new_head_mask)
        model.prune_heads(head_mask_dict)
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)
        # load trained model
        model = trainer.model

        # Compute metric and head importance again
        head_importance, preds, labels = compute_heads_importance(
            model, eval_dataloader, head_mask=new_head_mask
        )

    print(output_heatmap_mask)
    np.save(os.path.join(args.output_dir, f"output_heatmap_mask.npy"), output_heatmap_mask)
    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask

def numpy_to_dict(head_mask):
    out_dict = {}
    for i, layer in enumerate(head_mask):
        heads_to_prune = []
        for j, head in enumerate(layer):
            if head == 0:
                heads_to_prune.append(j)
        out_dict[i] = heads_to_prune
    return out_dict

def compute_score(preds, labels):
    total_count = 0
    correct_count = 0
    for prediction, label in zip(preds, labels):
        predicted_label = np.argmax(prediction)
        if int(label) == predicted_label:
            correct_count += 1
        total_count += 1
    return correct_count/total_count

def reset_model(args):
    model = ViTForImageClassification.from_pretrained(args.model_name)

    trainer = Trainer(model=model, args=args.training_args, data_collator=args.collate_fn, compute_metrics=args.compute_metrics,
        train_dataset=args.transformed_train_ds, eval_dataset=args.transformed_val_ds, tokenizer=args.feature_extractor, )

    processed_val_ds = trainer.get_eval_dataloader()

    # Distributed and parallel training
    model.to(args.device)
    return model, trainer, processed_val_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='Name of dataset to be loaded from huggingface datasets. Default is cifar100.')
    parser.add_argument('--output_dir', type=str, default='saved_numpys',
                        help='Output dir for the head masks')
    parser.add_argument('--dont_save_mask_all_iterations', action='store_true',
                        help='Call this to not save mask each iteration.')
    parser.add_argument('--masking_factor', type=int, default=0.2,
                        help='Prune masking_factor*remaining weights')
    parser.add_argument('--num_epochs', type=int, default=13,
                        help='Number of epochs to prune for.')
    parser.add_argument('--dont_normalize_importance_by_layer', action='store_true')
    parser.add_argument('--dont_normalize_global_importance', action='store_true')
    parser.add_argument('--model_name', default='starting_checkpoint/checkpoint-100',
                        help='Model name to load when rolling back weights')
    args = parser.parse_args()

    no_cuda = False
    local_rank = -1
    # Setup devices and distributed training
    if local_rank == -1 or no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = 0 if no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend
    try_masking = True
    # Load pretrained model
    # Initializing a ViT vit-base-patch16-224 style configuration
    if args.dataset_name == 'mnist':
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', image_mean=0.5,
                                                            image_std=0.5)
    elif args.dataset_name == 'cifar100':
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    train_ds, test_ds = load_dataset(args.dataset_name, split=['train', 'test'])
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    print(train_ds.info)
    print(train_ds.shape)

    print(feature_extractor)

    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        if args.dataset_name == 'mnist':
            inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
            inputs['pixel_values'] = torch.stack([inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values']],
                                                 dim=1)

            # Don't forget to include the labels!
            inputs['labels'] = example_batch['label']

        elif args.dataset_name == 'cifar100':
            inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

            # Don't forget to include the labels!
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

    ## CIFAR
    if args.dataset_name == 'cifar100': labels = train_ds.features['fine_label'].names

    # training_args = TrainingArguments(output_dir="./starting_checkpoint", per_device_train_batch_size=16,
    #     evaluation_strategy="no", max_steps=100, fp16=True, save_steps=100, eval_steps=100, logging_steps=100,
    #     learning_rate=2e-4, save_total_limit=2, remove_unused_columns=False, push_to_hub=False, report_to='tensorboard',
    #     load_best_model_at_end=False, disable_tqdm=True,)
    #
    # # Initial model. Train for 100 steps.
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(labels),
    #                                                   id2label={str(i): c for i, c in enumerate(labels)},
    #                                                   label2id={c: str(i) for i, c in enumerate(labels)})
    #
    # trainer = Trainer(model=model, args=training_args, data_collator=collate_fn, compute_metrics=compute_metrics,
    #     train_dataset=transformed_train_ds, eval_dataset=transformed_val_ds, tokenizer=feature_extractor, )
    #
    # # Distributed and parallel training
    # model.to(args.device)
    # train_results = trainer.train()
    # trainer.log_metrics("train", train_results.metrics)
    #
    # model.cpu()

    args.training_args = TrainingArguments(output_dir="./pruning_checkpoints", per_device_train_batch_size=16,
        evaluation_strategy="no", max_steps=1000, fp16=True, save_steps=1000, eval_steps=1000, logging_steps=100,
        learning_rate=2e-4, save_total_limit=3, remove_unused_columns=False, push_to_hub=False, report_to='tensorboard',
        load_best_model_at_end=False, disable_tqdm=True,)

    args.collate_fn = collate_fn
    args.compute_metrics = compute_metrics
    args.transformed_train_ds = transformed_train_ds
    args.traformed_val_ds = transformed_val_ds
    args.feature_extractor = feature_extractor

    head_mask = mask_heads(args)

    print(head_mask)

    # Head masks needs to be dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
    # When we finally have the mask, we can prune the heads and test performance speed gain.
    # model._prune_heads(head_mask)

if __name__ == "__main__":
    main()
