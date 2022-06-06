"""
This file contains masking head technique for ViT based on the LTH but by using attn * grad_attn. This is approach
differs because it uses an algorithm to decide what the next pruning percentage should be based on the validation loss.

This file was created by and designed by Christopher du Toit.
"""

import logging
import os
import argparse
import time
import sys

import numpy as np
from tqdm import tqdm
import torch

sys.path.append(r"C:\Users\Gebruiker\Documents\GitHub\transformers\src")
from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

logger = logging.getLogger(__name__)
# logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True
logging.getLogger().setLevel('CRITICAL')
logger.disabled = True


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
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=True)):
        input_ids = batch['pixel_values'].to(args.device)
        label_ids = batch['labels'].to(args.device)

        outputs = model(pixel_values=input_ids, labels=label_ids, output_attentions=True)

        loss, logits, attentions = (outputs.loss, outputs.logits, outputs.attentions)  # Loss and logits

        for attention_map in attentions:
            attention_map.retain_grad()
        loss.backward()
        attention_grads = [attention_map.grad for attention_map in attentions]

        all_heads_sensitivity = []
        for layer, layer_grad in zip(attentions, attention_grads):
            head_sensitivity = torch.zeros(layer.shape[1])
            tot_tokens = layer.shape[0]
            heads = layer.sum(dim=0)
            heads_grad = layer_grad.sum(dim=0)
            for idx, (head, head_grad) in enumerate(zip(heads, heads_grad)):
                head_sensitivity[idx] = (head * head_grad).abs().sum().detach().cpu() / (197 * 197)

            all_heads_sensitivity.append(head_sensitivity / tot_tokens)


        for attention_map, attention_grad_map in zip(attentions, attention_grads):
            attention_map.detach().cpu()
            attention_grad_map.detach().cpu()

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
        for layer_idx in range(n_layers):
            i_idx = 0
            for head_idx in range(n_heads):
                if head_mask[layer_idx][head_idx] == 1:
                    head_importance[layer_idx][head_idx] += all_heads_sensitivity[j_idx][i_idx]
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

    head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader)
    new_head_mask = torch.ones_like(head_importance)
    i = 0
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
            print(f"Iteration {i} head mask: \n {head_mask}")
            print(f"Iteration {i} head importance: \n {head_importance}")
        i += 1
        # heads from least important to most - keep only not-masked heads

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
            if head_importance.view(-1)[head.item()] > args.pruning_threshold:
                print(f"breaking: No more values below args.pruning_threshold")
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

        head_mask_dict = numpy_to_dict(new_head_mask)
        model.prune_heads(head_mask_dict)

        # Compute metric and head importance again
        head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, head_mask=new_head_mask)

    print(output_heatmap_mask)
    print(pruning_percentages)
    print(num_heads_pruned)
    print(total_num_heads_pruned)
    np.save(os.path.join(args.output_dir, f"output_heatmap_mask.npy"), output_heatmap_mask)
    pruning_percentages = np.array(pruning_percentages)
    num_heads_pruned = np.array(num_heads_pruned)
    total_num_heads_pruned = np.array(total_num_heads_pruned)
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
    # Initial model. Train for 100 steps.
    model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224', num_labels=len(args.labels),
                                                      id2label={str(i): c for i, c in enumerate(args.labels)},
                                                      label2id={c: str(i) for i, c in enumerate(args.labels)},
                                                      ignore_mismatched_sizes=True)

    trainer = Trainer(model=model, args=args.training_args, data_collator=args.collate_fn,
                      compute_metrics=args.compute_metrics, train_dataset=args.transformed_train_ds,
                      eval_dataset=args.transformed_val_ds, tokenizer=args.feature_extractor, )

    processed_val_ds = trainer.get_eval_dataloader()

    # Distributed and parallel training
    model.to(args.device)
    return model, trainer, processed_val_ds


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
    parser.add_argument('--pruning_threshold', type=float, default=0.8,
                        help='Value for which the head importance value should not be below.')
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
    if args.dataset_name == 'mnist':
        feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224', image_mean=0.5,
                                                                image_std=0.5)
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')

    train_ds, test_ds = load_dataset(args.dataset_name, split=['train', 'test'])
    splits = train_ds.train_test_split(test_size=0.1, shuffle=False)

    train_ds = splits['train']
    val_ds = splits['test']

    def transform(example_batch):
        """
        This method transformers the dataset applying the correct data augmentations to the dataset.
        """
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

    ## MNIST
    if args.dataset_name == 'mnist': args.labels = train_ds.features['label'].names

    ## CIFAR100
    if args.dataset_name == 'cifar100': args.labels = train_ds.features['fine_label'].names

    ## CIFAR10
    if args.dataset_name == 'cifar10': args.labels = train_ds.features['label'].names

    args.training_args = TrainingArguments(
        output_dir="./" + args.experiment_id + "/starting_checkpoint_" + args.dataset_name,
        per_device_train_batch_size=16, evaluation_strategy="no", max_steps=1000, fp16=True, save_steps=1000,
        eval_steps=1000, logging_steps=100, learning_rate=2e-4, save_total_limit=3, remove_unused_columns=False,
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
