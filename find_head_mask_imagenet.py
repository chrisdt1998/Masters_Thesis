"""
This file contains masking head technique for ViT based on the LTH but by using attn * grad_attn. This is approach
differs because it uses an algorithm to decide what the next pruning percentage should be based on the validation loss.

This file was created by and designed by Christopher du Toit.
"""

import logging
import os
import sys
import argparse
import time

import numpy as np
from tqdm import tqdm
import torch
sys.path.append(r"C:\Users\Gebruiker\Documents\GitHub\transformers\src")
from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments, AutoConfig
from datasets import load_dataset, load_metric
import datasets

logger = logging.getLogger(__name__)
# logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True
logging.getLogger().setLevel('CRITICAL')
logger.disabled = True
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path('D:\ImageNet')
datasets.config.HF_DATASETS_CACHE = Path('D:\ImageNet')


def compute_heads_importance(args, model, eval_dataloader, head_mask=None):
    """
    This method computes the head importance of the transformer heads in each layer. The way it does this is by
    computing attention sum(attn_head * attn_head_gradient)/num_tokens for each head and then dividing again by the
    number of heads in that layer. The attn_head and attn_head_gradient are computed by performing a forward and
    backward pass on the model, where the input is the validation dataset. We then normalize by both global
    and local (layer wise) normalization as this provides a significant boost in performance. The output values are the
    importance values for each head in the model. More information can be found in the accompanying report (TBD).
    :param args: Args parser containing arguments.
    :type args: argparse
    :param model: Model object for DeiT-B.
    :type model: class
    :param eval_dataloader: Evaluation dataloader containing the evaluation dataset.
    :type eval_dataloader: dataloader
    :param head_mask: Tensor containing the current head mask.
    :type head_mask: tensor
    :return: The head importance, predictions and true labels.
    """
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

    # Iterate through the evaluation data
    total_attn_maps = []
    total_attn_grad = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=True)):
        input_ids = batch['pixel_values'].to(args.device)
        label_ids = batch['labels'].to(args.device)

        outputs = model(pixel_values=input_ids, labels=label_ids, output_attentions=True)

        loss, logits, attentions = (outputs.loss, outputs.logits, outputs.attentions)  # Loss and logits

        for attention_map in attentions:
            if attention_map is not None:
                attention_map.retain_grad()
        loss.backward()
        attention_grads = [attention_map.grad if attention_map is not None else None for attention_map in attentions]

        if len(total_attn_maps) == 0:
            total_attn_maps = list(attentions)
            total_attn_grad = list(attention_grads)

        for layer_idx, (layer, layer_grad) in enumerate(zip(attentions, attention_grads)):
            if layer is not None:
                # print(total_attn_maps[layer_idx].shape, layer.shape)
                total_attn_maps[layer_idx] += layer
                total_attn_grad[layer_idx] += layer_grad

        for attention_map, attention_grad_map in zip(attentions, attention_grads):
            if attention_map is not None:
                attention_map.detach().cpu()
                attention_grad_map.detach().cpu()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

    j_idx = 0
    for layer_idx in range(n_layers):
        i_idx = 0
        if total_attn_maps[j_idx] is not None:
            x = (total_attn_maps[j_idx] * total_attn_grad[j_idx]).sum(dim=0).abs()
            for head_idx in range(n_heads):
                if head_mask[layer_idx][head_idx] == 1:
                    head_importance[layer_idx][head_idx] = x[i_idx].sum().detach().cpu() / (197 * 197 * 8)
                    # head_importance[layer_idx][head_idx] += all_heads_sensitivity[j_idx][i_idx]
                    i_idx += 1
        j_idx += 1

    # Global importance normalization.
    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (
                head_importance.max() - head_importance.min())

    # Layerwise importance normalization.
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # Print/save matrices
    np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

    return head_importance, preds, labels


def mask_heads(args):
    """
    This method applies the masking based on a masking factor which is iteratively computed each iteration based on the
    previous validation result. The masking procedure masks the heads in terms of importance computed via the
    compute_heads_importance method, with the least importance heads being pruned first. The pruning is done globally,
    i.e. some layers can be completely removed.
    :param args: Args parser containing arguments.
    :type args: argparse
    :return: Final head mask
    :rtype: tensor
    """
    # Initialize the model and trainer
    model, trainer, eval_dataloader = reset_model(args)

    # Train the model for 1000 steps (about 33% of the training data).
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)

    # load trained model
    model = trainer.model

    head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader)
    original_score = compute_score(preds, labels)
    current_score = original_score
    print(f"Starting score {original_score}")
    new_head_mask = torch.ones_like(head_importance)
    i = args.starting_epoch
    final_scores = []
    pruning_percentages = [0, ]
    num_heads_pruned = [0, ]
    total_num_heads_pruned = [0, ]
    masking_factor = args.initial_pruning_factor
    output_heatmap_mask = np.ones((12, 12)) * args.num_epochs
    while i < args.num_epochs:
        head_mask = new_head_mask.clone()  # save current head mask
        if not args.dont_save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
            final_scores.append(current_score)
            print(f"Iteration {i} score: {current_score}")
            print(f"Iteration {i} head mask: \n {head_mask}")
            print(f"Iteration {i} head importance: \n {head_importance}")
        i += 1
        # heads from least important to most - keep only not-masked heads
        masking_factor = compute_next_pruning_factor(args, original_score, current_score, masking_factor)
        masking_factor = min(masking_factor, args.initial_pruning_factor)

        # If masking factor is less than 0, this means that we should backtrack
        if masking_factor < 0:
            print(f"Masking factor, {masking_factor} is less than 0, therefore we need to backtrack.")
            previous_masking_factor = pruning_percentages[-1]
            masking_factor = previous_masking_factor + masking_factor
            head_importance = torch.from_numpy(np.load(args.output_dir + f"/head_importance_{i-2}.npy")).to(args.device)
            new_head_mask = torch.from_numpy(np.load(args.output_dir + f"/head_mask_{i-2}.npy"))

        # If masking factor is still less than 0 despite backtracking
        if masking_factor < 0:
            print(f"Masking factor still negative. Only pruning 1 head.")
            masking_factor = 1/144

        if i == args.num_epochs:
            break

        unpruned_heads = torch.sum(new_head_mask)
        pruning_percentages.append(masking_factor)
        num_to_mask = max(1, round(144 * masking_factor))
        total_num_heads_pruned.append(144 - unpruned_heads.item() + num_to_mask)
        num_heads_pruned.append(num_to_mask)

        print(
            f"num_to_mask for iter {i} is {num_to_mask}, total pruned already is {144 - unpruned_heads}, ({(144 - unpruned_heads) * 100 / 144}%)")
        print(f"Masking factor is {masking_factor}")

        # Here we choose whether or not to prune whole layers. Pruning whole layers requires pruning the corresponding
        # MLPs too.
        if not args.prune_whole_layers:
            for layer_idx, layer in enumerate(head_mask):
                if torch.sum(layer) == 1:
                    head_importance[layer_idx] = float("Inf")
        head_importance[head_mask == 0.0] = float("Inf")


        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            print("Break 1", len(current_heads_to_mask), num_to_mask)
            break

        if unpruned_heads - num_to_mask < 1:
            num_to_mask = unpruned_heads - 1
            print(f"Number of heads left <= 1 therfore, new num_to_mask is: {num_to_mask}")
        print(f"Number of heads left: {unpruned_heads - num_to_mask}")
        if torch.sum(new_head_mask) < 5:
            print(f"Only {torch.sum(new_head_mask)} number of heads left causing the algorithm to stop searching.")
            break

        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            print(head_importance.view(-1)[head.item()])

            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                print("Break 2", len(selected_heads_to_mask), num_to_mask)
                break
            layer_idx = head.item() // 12
            head_idx = head.item() % 12
            if not args.prune_whole_layers:
                if torch.sum(new_head_mask[layer_idx]) != 1:
                    new_head_mask[layer_idx][head_idx] = 0.0
                    output_heatmap_mask[layer_idx][head_idx] = i
                    selected_heads_to_mask.append(head.item())
                    print(
                        f"Layer {layer_idx}, head number {head_idx} is pruned. Number of heads pruned is: {len(selected_heads_to_mask)}")
            else:
                new_head_mask[layer_idx][head_idx] = 0.0
                output_heatmap_mask[layer_idx][head_idx] = i
                selected_heads_to_mask.append(head.item())
                print(
                    f"Layer {layer_idx}, head number {head_idx} is pruned. Number of heads pruned is: {len(selected_heads_to_mask)}")

        if not selected_heads_to_mask:
            print("No more prunable heads available.", selected_heads_to_mask)
            break


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
        head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, head_mask=new_head_mask)
        current_score = compute_score(preds, labels)

    print(output_heatmap_mask)
    print(final_scores)
    print(pruning_percentages)
    print(num_heads_pruned)
    print(total_num_heads_pruned)
    np.save(os.path.join(args.output_dir, f"output_heatmap_mask.npy"), output_heatmap_mask)
    final_scores = np.array(final_scores)
    pruning_percentages = np.array(pruning_percentages)
    num_heads_pruned = np.array(num_heads_pruned)
    total_num_heads_pruned = np.array(total_num_heads_pruned)
    np.save(os.path.join(args.output_dir, f"final_scores.npy"), final_scores)
    np.save(os.path.join(args.output_dir, f"pruning_percentages.npy"), pruning_percentages)
    np.save(os.path.join(args.output_dir, f"num_heads_pruned.npy"), num_heads_pruned)
    np.save(os.path.join(args.output_dir, f"total_num_heads_pruned.npy"), total_num_heads_pruned)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


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


def compute_score(preds, labels):
    """
    This method computes the score of the validation dataset.
    :param preds: Predicted labels.
    :type preds: nd.array
    :param labels: True labels.
    :type labels: nd.array
    :return: The percentage of correct predictions.
    :rtype: float
    """
    total_count = 0
    correct_count = 0
    for prediction, label in zip(preds, labels):
        predicted_label = np.argmax(prediction)
        if int(label) == predicted_label:
            correct_count += 1
        total_count += 1
    return correct_count / total_count


def reset_model(args):
    """
    This method resets the model to an earlier point in training after each iteration as done in the Lottery Ticket
    Hypothesis.
    :param args: Args parser containing arguments.
    :type args: argparse
    :return: Model, model trainer and validation dataloader.
    :rtype: tuple
    """
    model = ViTForImageClassification.from_pretrained(
        args.experiment_id + "/starting_checkpoint_" + args.dataset_name + "/checkpoint-1000")

    trainer = Trainer(model=model, args=args.training_args, data_collator=args.collate_fn,
                      compute_metrics=args.compute_metrics, train_dataset=args.transformed_train_ds,
                      eval_dataset=args.transformed_val_ds, tokenizer=args.feature_extractor, )

    processed_val_ds = trainer.get_eval_dataloader()

    # Distributed and parallel training
    model.to(args.device)
    return model, trainer, processed_val_ds


def compute_next_pruning_factor(args, original_score, current_score, current_pruning_factor):
    """
    This method contains the formula for computing the next pruning iteration.
    :param args: Args parser containing arguments.
    :type args: argparse
    :param original_score: The original validation score without any pruning.
    :type original_score: float
    :param current_score: The current score with the new pruning.
    :type current_score: float
    :param current_pruning_factor: The current pruning factor.
    :type current_pruning_factor: float
    :return: The next pruning factor.
    :rtype: float
    """
    score_difference = (original_score - current_score)/original_score
    pruning_threshold = 1 - args.pruning_threshold
    # pruning_factor = args.initial_pruning_factor * (pruning_threshold - score_difference)/pruning_threshold
    pruning_factor = current_pruning_factor * (pruning_threshold - score_difference) / pruning_threshold
    return pruning_factor


def main():
    """
    This is the main function to run. The parameters for the function can be added to the argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='Name of dataset to be loaded from huggingface datasets. Default is cifar100.')
    # parser.add_argument('--output_dir', type=str, default='saved_numpys',
    #                     help='Output dir for the head masks')
    parser.add_argument('--dont_save_mask_all_iterations', action='store_true',
                        help='Call this to not save mask each iteration.')
    parser.add_argument('--initial_pruning_factor', type=float, default=0.2, help='Prune pruning_factor*remaining weights')
    parser.add_argument('--pruning_threshold', type=float, default=0.92,
                        help='Factor of the original validation accuracy that the next validation accuracy should not go below.')
    parser.add_argument('--dont_normalize_importance_by_layer', action='store_true')
    parser.add_argument('--dont_normalize_global_importance', action='store_true')
    parser.add_argument('--model_name', default='starting_checkpoint/checkpoint-100',
                        help='Model name to load when rolling back weights')
    parser.add_argument('--iteration_id', type=str, default='base_run',
                        help='Iteration id to be able to run multiple scripts at the same time and identify.')
    parser.add_argument('--experiment_id', type=str, default='experiment_1',
                        help='Experiment id for the experiment we are currently running.')
    parser.add_argument('--prune_whole_layers', action='store_true',
                        help='Call this if we want to be able to prune whole layers and the corresponding MLPs (TBD).')
    parser.add_argument('--starting_epoch', type=int, default=0, help='Choose epoch number to start from.')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.experiment_id, 'saved_numpys_' + args.iteration_id)
    # args.output_dir = args.experiment_id + '/saved_numpys_' + args.iteration_id
    # args.num_epochs = int(100/args.initial_masking_factor) + 1
    args.num_epochs = 9

    print(f"This is experiment {args.experiment_id} with iteration id {args.iteration_id}.")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    no_cuda = False
    local_rank = -1
    # Setup devices and distributed training
    if local_rank == -1 or no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = 0 if no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Load pretrained model
    # Initializing a ViT vit-base-patch16-224 style configuration
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')

    train_ds, test_ds, val_ds = load_dataset("imagenet-1k", use_auth_token="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",
                                             split=['train', 'test', 'validation'])

    def transform(example_batch):
        """
        This method transformers the dataset applying the correct data augmentations to the dataset.
        """
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['label']

        return inputs

    transformed_train_ds = train_ds.with_transform(transform)
    # transformed_test_ds = test_ds.with_transform(transform)
    transformed_val_ds = val_ds.with_transform(transform)

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

    labels = train_ds.features['label'].names


    if not os.path.isdir("./" + args.experiment_id + "/starting_checkpoint_" + args.dataset_name):
        print(f"Starting checkpoint does not exist. Creating one now.")

        training_args = TrainingArguments(output_dir="./" + args.experiment_id + "/starting_checkpoint_" + args.dataset_name,
                                          per_device_train_batch_size=16, evaluation_strategy="no", max_steps=1000,
                                          fp16=True, save_steps=1000, eval_steps=1000, logging_steps=1000, learning_rate=2e-4,
                                          save_total_limit=2, remove_unused_columns=False, push_to_hub=False,
                                          report_to='tensorboard', load_best_model_at_end=False, disable_tqdm=True,
                                          log_level='critical', )

        # # Initial model. Train for 100 steps.
        # model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224', num_labels=len(labels),
        #                                                   id2label={str(i): c for i, c in enumerate(labels)},
        #                                                   label2id={c: str(i) for i, c in enumerate(labels)},
        #                                                   ignore_mismatched_sizes=True)

        model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
        for i in range(12):
            model.vit.encoder.layer[i].apply(model._init_weights)

        trainer = Trainer(model=model, args=training_args, data_collator=collate_fn, compute_metrics=compute_metrics,
                          train_dataset=transformed_train_ds, eval_dataset=transformed_val_ds,
                          tokenizer=feature_extractor, )

        # Distributed and parallel training
        model.to(args.device)
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)

        model.cpu()

    args.training_args = TrainingArguments(
        output_dir="./" + args.experiment_id + "/pruning_checkpoints_" + args.dataset_name + '_' + args.iteration_id,
        per_device_train_batch_size=16, evaluation_strategy="no", max_steps=10000, fp16=True, save_steps=10000,
        eval_steps=10000, logging_steps=1000, learning_rate=6.25e-8, save_total_limit=3, remove_unused_columns=False,
        push_to_hub=False, report_to='tensorboard', load_best_model_at_end=False, disable_tqdm=True,
        log_level='critical', )

    args.collate_fn = collate_fn
    args.compute_metrics = compute_metrics
    args.transformed_train_ds = transformed_train_ds
    args.transformed_val_ds = transformed_val_ds
    args.feature_extractor = feature_extractor

    start_time = time.time()
    head_mask = mask_heads(args)
    print(f"Time taken = {time.time() - start_time}")
    print(f"This was experiment {args.experiment_id} with iteration id {args.iteration_id}.")

    print(head_mask)


if __name__ == "__main__":
    main()
