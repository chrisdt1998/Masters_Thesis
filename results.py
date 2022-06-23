"""
This folder contains code to plot the results from different head pruning experiments.

This file was created by and designed by Christopher du Toit.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
# print(plt.rcParams["figure.figsize"])
# plt.rcParams["figure.figsize"] = (8, 6)

def head_mask_3_results():
    results = [0.8782, 0.8484, 0.8368, 0.8116, 0.7142, 0.3514, 0.2026, 0.1616, 0.1264]
    results = np.array(results)
    percentages = np.arange(0, 90, 10)

    plt.plot(percentages, results)
    plt.title("mask_head_vit_3_cifar100")
    plt.xlabel("Percentage of heads pruned")
    plt.ylabel("Validation accuracy of the model.")
    plt.show()


def head_mask_2_results():
    results = [0.8784, 0.8680, 0.8522, 0.8322, 0.8230, 0.7582, 0.6882, 0.4556, 0.3154, 0.194]
    results = np.array(results)
    percentages = np.arange(0, 100, 10)

    plt.plot(percentages, results)
    plt.title("mask_head_vit_2_cifar100")
    plt.xlabel("Percentage of heads pruned")
    plt.ylabel("Validation accuracy of the model.")
    plt.show()


def produce_heat_map(arr, title):
    fig, ax = plt.subplots()
    im = ax.imshow(arr)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_yticks(np.arange(arr.shape[0]))


    # Loop over data dimensions and create text annotations.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            text = ax.text(j, i, arr[i][j], ha="center", va="center", color="w")

    ax.set_title(title)
    ax.set_xlabel("Head number")
    ax.set_ylabel("Layer number")
    fig.tight_layout()
    plt.show()


def head_mask_2_iter_mnist():
    heatmap = np.load('saved_numpys_mnist_2_iter/output_heatmap_mask.npy')
    produce_heat_map(heatmap, 'MNIST pruning heatmap')


def prune_percentages_arr(num_iters):
    pruned = 0
    unpruned = 100
    percentages = []
    percentages.append(pruned)
    for i in range(num_iters):
        unpruned = unpruned*0.8
        pruned = 100 - unpruned
        percentages.append(round(pruned, 1))
    return percentages


def head_mask_2_iter_comparison():
    heatmap = np.load('saved_numpys_cifar100_2_iter/output_heatmap_mask.npy')
    produce_heat_map(heatmap, 'CIFAR100 pruning heatmap')
    heatmap = np.load('saved_numpys_cifar10_2_iter/output_heatmap_mask.npy')
    produce_heat_map(heatmap, 'CIFAR10 pruning heatmap')
    heatmap = np.load('saved_numpys_mnist_2_iter/output_heatmap_mask.npy')
    produce_heat_map(heatmap, 'MNIST pruning heatmap')
    percentages = prune_percentages_arr(12)
    x = np.arange(13)
    results_cifar100 = np.load('saved_numpys_cifar100_2_iter/final_scores.npy')
    results_cifar10 = np.load('saved_numpys_cifar10_2_iter/final_scores.npy')
    results_mnist = np.load('saved_numpys_mnist_2_iter/final_scores.npy')

    plt.plot(x, results_cifar100, label='CIFAR100', marker='x')
    plt.plot(x, results_cifar10, label='CIFAR10', marker='x')
    plt.plot(x, results_mnist, label='MNIST', marker='x')
    plt.title("mask_head_vit_2_iter")
    plt.xlabel("Percentage of heads pruned. %")
    plt.ylabel("Validation accuracy of the model. %")

    plt.xticks(x, labels=percentages)
    plt.legend()
    plt.show()


def plot_heatmaps():
    heatmap_cifar100 = np.load('saved_numpys_cifar100_2_iter/output_heatmap_mask.npy')
    heatmap_cifar10 = np.load('saved_numpys_cifar10_2_iter/output_heatmap_mask.npy')
    heatmap_mnist = np.load('saved_numpys_mnist_2_iter/output_heatmap_mask.npy')
    heatmap = np.zeros((12, 12))
    for i in range(heatmap_cifar100.shape[0]):
        for j in range(heatmap_cifar100.shape[1]):
            if heatmap_cifar100[i][j] == heatmap_cifar10[i][j] == heatmap_mnist[i][j]:
                heatmap[i][j] = heatmap_cifar100[i][j]

    produce_heat_map(heatmap, 'Comparison heatmap all datasets')
    heatmap = np.zeros((12, 12))
    for i in range(heatmap_cifar100.shape[0]):
        for j in range(heatmap_cifar100.shape[1]):
            if heatmap_cifar100[i][j] == heatmap_cifar10[i][j]:
                heatmap[i][j] = heatmap_cifar100[i][j]

    produce_heat_map(heatmap, 'Comparison heatmap CIFAR100 and CIFAR10')
    heatmap = np.zeros((12, 12))
    for i in range(heatmap_cifar100.shape[0]):
        for j in range(heatmap_cifar100.shape[1]):
            if heatmap_cifar100[i][j] == heatmap_mnist[i][j]:
                heatmap[i][j] = heatmap_cifar100[i][j]

    produce_heat_map(heatmap, 'Comparison heatmap CIFAR100 and MNIST')
    heatmap = np.zeros((12, 12))
    for i in range(heatmap_cifar100.shape[0]):
        for j in range(heatmap_cifar100.shape[1]):
            if heatmap_cifar10[i][j] == heatmap_mnist[i][j]:
                heatmap[i][j] = heatmap_cifar10[i][j]

    produce_heat_map(heatmap, 'Comparison heatmap CIFAR10 and MNIST')


def cifar100_full_trained():
    training_times = np.load('cifar100_fully_trained_test_results/training_times.npy')
    testing_samples_per_second= np.load('cifar100_fully_trained_test_results/testing_samples_per_second.npy')
    accuracy = np.load('cifar100_fully_trained_test_results/accuracy.npy')
    results_cifar100 = np.load('saved_numpys_cifar100_2_iter/final_scores.npy')
    accuracy_attempt7 = np.load('cifar100_fully_trained_test_results/accuracy_attempt7.npy')
    results_cifar100_attempt7 = np.load('saved_numpys_attempt7_rerun/final_scores.npy')
    x = np.arange(13)
    percentages = prune_percentages_arr(12)
    results_cifar100 *= 100
    results_cifar100_attempt7 *= 100
    print(f"accuracy \n {accuracy} \n validation results \n {results_cifar100}")
    print(f"accuracy \n {accuracy_attempt7} \n validation results \n {results_cifar100_attempt7}")
    print(accuracy_attempt7 - results_cifar100_attempt7)
    plt.plot(x, results_cifar100, 'b--', label='Validation old', marker='x')
    plt.plot(x, accuracy, 'b', label='Fully trained old', marker='x')
    plt.plot(x, results_cifar100_attempt7, 'r--', label='Validation new', marker='x')
    plt.plot(x, accuracy_attempt7, 'r', label='Fully trained new', marker='x')
    plt.title("CIFAR100 fully trained vs validation accuracy")
    plt.xlabel("Percentage of heads pruned. %")
    plt.ylabel("Accuracy of the model. %")

    plt.xticks(x, labels=percentages)
    plt.legend()
    plt.show()


def cifar100_smart_iter():
    pruning_percentages = np.load('saved_numpys_iterative_test_1/pruning_percentages.npy')
    validation_results = np.load('saved_numpys_iterative_test_1/final_scores.npy') * 100
    full_trained_results = np.load('saved_numpys_iterative_test_1/full_trained_results.npy')
    print(pruning_percentages)
    x = np.arange(pruning_percentages.shape[0])
    for i, percent in enumerate(pruning_percentages):
        if percent < 0:
            pruning_percentages[i] = 1/144

    pruning_percentages = np.round(pruning_percentages, 3)
    print(pruning_percentages)
    pruning_percentages = np.cumsum(pruning_percentages)
    print(pruning_percentages)
    plt.xticks(x, labels=pruning_percentages)
    plt.plot(x, validation_results, marker='x', linestyle='-', label='Validation scores', color='r')
    plt.plot(x, full_trained_results, marker='x', linestyle='-', label='Full trained scores', color='b')
    plt.axhline(y=0.92 * validation_results[0], color='r', linestyle='--')
    plt.axhline(y=full_trained_results[0] - 3, color='b', linestyle='--')

    plt.show()


def fix_pruning_percentages(percentages):
    print(percentages)
    for i, percent in enumerate(percentages):
        if percent < 0:
            percentages[i] = 1/144
    return percentages


# COMPLETED PLOTTED
def normalizing_testing():
    no_norm = np.load('saved_numpys_deit_no_normalization/final_scores.npy')
    layer_norm = np.load('saved_numpys_deit_normalize_layers_only/final_scores.npy')
    global_norm = np.load('saved_numpys_deit_normalize_global_only/final_scores.npy')
    both_norms_layer_first = np.load('saved_numpys_deit_both_normalizations/final_scores.npy')
    both_norms_global_first = np.load('saved_numpys_deit_global_first_both_normalizations/final_scores.npy')

    percentages_no_norm = np.cumsum(np.load('saved_numpys_deit_no_normalization/pruning_percentages.npy'))
    percentages_layer_norm = np.cumsum(np.load('saved_numpys_deit_normalize_layers_only/pruning_percentages.npy'))
    percentages_global_norm = np.cumsum(np.load('saved_numpys_deit_normalize_global_only/pruning_percentages.npy'))
    percentages_both_norm_layer_first = np.cumsum(np.load('saved_numpys_deit_both_normalizations/pruning_percentages.npy'))
    percentages_both_norm_global_first = np.cumsum(np.load('saved_numpys_deit_global_first_both_normalizations/pruning_percentages.npy'))

    # print(no_norm)
    # print(percentages_no_norm)
    # print(np.load('saved_numpys_deit_no_normalization/pruning_percentages.npy'))
    # num_heads_pruned = np.load('saved_numpys_deit_no_normalization/num_heads_pruned.npy')
    # total_num_heads_pruned = np.load('saved_numpys_deit_no_normalization/total_num_heads_pruned.npy')
    # print(num_heads_pruned)
    # print(total_num_heads_pruned)
    # exit(1)

    plt.plot(percentages_no_norm[:9], no_norm, label='No normalization', marker='x')
    plt.plot(percentages_layer_norm[:9], layer_norm, label='Layer normalization', marker='x')
    plt.plot(percentages_global_norm[:9], global_norm, label='Global normalization', marker='x')
    plt.plot(percentages_both_norm_layer_first[:9], both_norms_layer_first, label='Both, layer first.', marker='x')
    plt.plot(percentages_both_norm_global_first, both_norms_global_first,
             label='Both, global first.', marker='x')

    plt.xlabel('Factor of heads pruned.')
    plt.ylabel('Validation score.')
    plt.title('Comparison of the different normalizations.')
    plt.legend()
    plt.show()


# COMPLETED PLOTTED
def experiments_threshold_cifar100():
    plt.clf()
    # thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    thresholds = np.array([0.90, 0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    # validation_scores = np.array([0.7312, 0.7484, 0.7686, 0.788, 0.802, 0.8142, 0.83, 0.8252]) * 100
    validation_scores = np.array([0.7484, 0.788, 0.802, 0.8142, 0.83, 0.8252]) * 100
    # num_heads_pruned = np.array([101, 100, 92, 92, 76, 60, 19, 45])
    num_heads_pruned = np.array([100, 92, 76, 60, 19, 45])
    percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.8318, 0.83, 0.8496, 0.8517, 0.8543, 0.8601, 0.866, 0.865]) * 100
    fully_trained_scores = np.array([0.83, 0.8517, 0.8543, 0.8601, 0.866, 0.865]) * 100
    plt.plot(0, 87.06, marker='x', label='Unpruned')
    plt.annotate(87.06, (0, 87.06), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - 87.06

    plt.plot(percentage_pruned, validation_scores, marker='x', label='Val')
    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')
    for i, (x, y) in enumerate(zip(percentage_pruned, validation_scores)):
        label = "{:.1f}".format(thresholds[i])
        if i == 0:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(7, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        else:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.1f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])
        if i == 0:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(5, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        elif i == 1:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(-5, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        else:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for cifar100")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower left')
    plt.show()


# COMPLETED PLOTTTED
def experiments_threshold_cifar100_one_train(with_random_init, with_pretrained_init):
    plt.clf()
    # thresholds = (np.array([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]) * 100).astype(int)
    # validation_scores = np.array([0.5548, 0.5684, 0.6140, 0.65, 0.7496, 0.756, 0.77]) * 100
    # num_heads_pruned = np.array([89, 86, 82, 75, 68, 61, 48])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.8277, 0.8378, 0.8437, 0.8427, 0.8496, 0.8493, 0.8534]) * 100
    # fully_trained_scores_with_pretrained = np.array([0.8485, 0.8531, 0.8567, 0.8568, 0.8593, 0.8556, 0.8656]) * 100
    thresholds = (np.array([0.65, 0.75, 0.80, 0.85, 0.90, 0.95]) * 100).astype(int)
    validation_scores = np.array([0.5548, 0.6140, 0.65, 0.7496, 0.756, 0.77]) * 100
    num_heads_pruned = np.array([89, 82, 75, 68, 61, 48])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.8277, 0.8437, 0.8427, 0.8496, 0.8493, 0.8534]) * 100
    fully_trained_scores_with_pretrained = np.array([0.8485, 0.8567, 0.8568, 0.8593, 0.8556, 0.8656]) * 100
    plt.plot(0, 87.06, marker='x', label='Unpruned')
    plt.annotate(87.06, (0, 87.06), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - 87.06
    difference_scores_with_pretrained = fully_trained_scores_with_pretrained - 87.06

    plt.plot(percentage_pruned, validation_scores, marker='x', label='Val')

    for i, (x, y) in enumerate(zip(percentage_pruned, validation_scores)):
        label = "{:.0f}".format(thresholds[i])
        coords = [
            [10, 5],
            [10, 5],
            [10, 5],
            [10, 5],
            [0, 10],
            [0, 10],
        ]
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    if with_random_init:
        plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full, random initialization')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 50, 89.50))
        for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
            label = "{:.0f}".format(thresholds[i])
            label_diff = "{:.1f}".format(difference_scores[i])

            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    if with_pretrained_init:
        plt.plot(percentage_pruned, fully_trained_scores_with_pretrained, marker='x', label='Full, with pretrained')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 50, 89.50))
        for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores_with_pretrained)):
            label = "{:.0f}".format(thresholds[i])
            label_diff = "{:.1f}".format(difference_scores_with_pretrained[i])

            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for cifar100 One Train")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def experiments_threshold_cifar100_prune_and_train():
    plt.clf()
    thresholds = np.array([0.94, 0.96, 0.97, 0.98, 0.99]) * 100
    num_heads_pruned = np.array([91, 77, 64, 49, 42])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.7846, 0.8491, 0.8116, 0.8603, 0.8615]) * 100
    plt.plot(0, 87.06, marker='x', label='Unpruned')
    plt.annotate(87.06, (0, 87.06), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - 87.06

    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')
    # x1, x2, y1, y2 = plt.axis()
    plt.axis((-5, 70, 77, 88))

    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.0f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])

        coords = [
            [5, 10, 0, -15],
            [0, 10, -3, -30],
            [0, 15, 0, -15],
            [0, 10, -3, -25],
            [0, 10, 0, -15]
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    plt.title("Experiment: Pruning thresholds for cifar100 Train and Prune")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower left')
    plt.show()


def experiments_threshold_cifar100_no_train():
    plt.clf()
    thresholds = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0]) * 100
    num_heads_pruned = np.array([101, 100, 92, 92, 76, 60, 19, 45])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.8318, 0.83, 0.8496, 0.8517, 0.8543, 0.8601, 0.866, 0.865]) * 100
    plt.plot(0, 87.06, marker='x', label='Unpruned')
    plt.annotate(87.06, (0, 87.06), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - 87.06

    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')


    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.1f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, -15),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for cifar100")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# COMPLETED PLOTTED
def experiments_threshold_cifar10_limited():
    plt.clf()
    # thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    # validation_scores = np.array([0.8544, 0.8808, 0.8868, 0.9226, 0.927, 0.9458, 0.9566, 0.9642]) * 100
    # num_heads_pruned = np.array([129, 129, 124, 115, 110, 94, 71, 33])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.8841, 0.9185, 0.9362, 0.9498, 0.9526, 0.9545, 0.9624, 0.9681]) * 100
    thresholds = np.array([0.90, 0.92, 0.96, 0.98, 0.99, 0.999]) * 100
    validation_scores = np.array([0.8808, 0.8868, 0.927, 0.9458, 0.9566, 0.9642]) * 100
    num_heads_pruned = np.array([129, 124, 110, 94, 71, 33])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.9185, 0.9362, 0.9526, 0.9545, 0.9624, 0.9681]) * 100
    unpruned_score = 96.94
    plt.plot(0, unpruned_score, marker='x', label='Unpruned')
    plt.annotate(unpruned_score, (0, unpruned_score), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - unpruned_score

    plt.plot(percentage_pruned, validation_scores, marker='x', label='Val')
    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')
    for i, (x, y) in enumerate(zip(percentage_pruned, validation_scores)):
        label = "{:.1f}".format(thresholds[i])
        if i == 0 or i == 1:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(-15, -8),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        else:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(-9, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.1f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])

        coords = [
            [10, 7, -7, -15],
            [8, 7, -10, -12],
            [5, 10, -5, -15],
            [0, 8, 0, -15],
            [0, 8, 0, -12],
            [10, 8, -12, -10],
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for cifar10 limited")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower left')
    plt.show()


# COMPLETED PLOTTED
def experiments_threshold_cifar10_global_pruning():
    plt.clf()
    # thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    # validation_scores = np.array([0.841, 0.8618, 0.875, 0.9224, 0.9396, 0.9498, 0.9586, 0.9634]) * 100
    # num_heads_pruned = np.array([127, 127, 124, 115, 112, 86, 79, 16])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.8955, 0.8895, 0.918, 0.9238, 0.9529, 0.9635, 0.9666, 0.9709]) * 100
    thresholds = np.array([0.90, 0.92, 0.96, 0.98, 0.99, 0.999]) * 100
    validation_scores = np.array([0.8618, 0.875, 0.9396, 0.9498, 0.9586, 0.9634]) * 100
    num_heads_pruned = np.array([127, 124, 112, 86, 79, 16])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.8895, 0.918, 0.9529, 0.9635, 0.9666, 0.9709]) * 100
    unpruned_score = 96.94
    plt.plot(0, unpruned_score, marker='x', label='Unpruned')
    plt.annotate(unpruned_score, (0, unpruned_score), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - unpruned_score

    plt.plot(percentage_pruned, validation_scores, marker='x', label='Val')
    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 85, 98))
    for i, (x, y) in enumerate(zip(percentage_pruned, validation_scores)):
        label = "{:.1f}".format(thresholds[i])
        coords = [
            [-15, -7],
            [-15, -7],
            [-15, -10],
            [0, -15],
            [-3, -15],
            [0, -15],
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.1f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])
        coords = [
            [10, 7, 0, -15],
            [8, 7, -8, -12],
            [3, 10, -5, -15],
            [3, 8, 3, -15],
            [-3, 8, -4, -12],
            [0, 8, 0, -12],
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for cifar10")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# COMPLETED PLOTTED
def experiments_threshold_cifar10_one_train():
    plt.clf()
    # thresholds = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.999]) * 100
    # validation_scores = np.array([0.4884, 0.5356, 0.6632, 0.7404, 0.8564, 0.9174, 0.9634]) * 100
    # num_heads_pruned = np.array([111, 110, 106, 101, 95, 86, 33])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.9305, 0.9275, 0.9555, 0.9577, 0.9632, 0.9545, 0.9707]) * 100
    thresholds = np.array([0.45, 0.65, 0.85, 0.95, 0.999]) * 100
    validation_scores = np.array([0.4884, 0.6632, 0.8564, 0.9174, 0.9634]) * 100
    num_heads_pruned = np.array([111, 106, 95, 86, 33])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.9305, 0.9555, 0.9632, 0.9545, 0.9707]) * 100
    plt.plot(0, 96.94, marker='x', label='Unpruned')
    plt.annotate(96.94, (0, 96.94), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - 96.94

    plt.plot(percentage_pruned, validation_scores, marker='x', label='Val')
    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 45, 101))
    for i, (x, y) in enumerate(zip(percentage_pruned, validation_scores)):
        label = "{:.1f}".format(thresholds[i])
        coords = [
            [-15, -5],
            [-15, -5],
            [-15, -5],
            [-5, -15],
            [0, -15]

        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.1f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])
        coords = [
            [5, 10, 5, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [15, 8, -15, 8]
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    plt.title("Experiment: Pruning thresholds for cifar10 One Train")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# COMPLETED PLOTTED
def experiments_threshold_cifar10_train_and_prune():
    plt.clf()
    # thresholds = np.array([0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    # num_heads_pruned = np.array([122, 116, 92, 62, 0])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.9355, 0.9456, 0.9602, 0.9657, 0.9693]) * 100
    thresholds = np.array([0.94, 0.96, 0.98, 0.99]) * 100
    num_heads_pruned = np.array([122, 116, 92, 62])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.9355, 0.9456, 0.9602, 0.9657]) * 100
    plt.plot(0, 96.94, marker='x', label='Unpruned')
    plt.annotate(96.94, (0, 96.94), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - 96.94

    plt.plot(percentage_pruned, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 93, 97.3))

    for i, (x, y) in enumerate(zip(percentage_pruned, fully_trained_scores)):
        label = "{:.0f}".format(thresholds[i])
        label_diff = "{:.1f}".format(difference_scores[i])

        coords = [
            [10, -3, -15, -3],
            [10, -3, -15, -3],
            [12, 5, -15, -8],
            [0, 10, 0, -15]
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    plt.title("Experiment: Pruning thresholds for cifar10 Train and Prune")
    plt.xlabel("Percentage of heads pruned.")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower left')
    plt.show()


# COMPLETED PLOTTED
def experiments_threshold_mnist_limited():
    plt.clf()
    # thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    # validation_scores = np.array([0.9921, 0.9922, 0.9925, 0.992, 0.9896, 0.992, 0.9905, 0.9948]) * 100
    # num_heads_pruned = np.array([132, 132, 132, 132, 132, 132, 132.0, 90])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.995, 0.9956, 0.9954, 0.9948, 0.9953, 0.9955, 0.9943, 0.9964]) * 100
    thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]) * 100
    validation_scores = np.array([0.9921, 0.9922, 0.9925, 0.992, 0.9896, 0.992, 0.9905]) * 100
    num_heads_pruned = np.array([132, 132, 132, 132, 132, 132, 132.0])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.995, 0.9956, 0.9954, 0.9948, 0.9953, 0.9955, 0.9943]) * 100
    unpruned_score = 99.40
    plt.plot(100, unpruned_score, marker='x', label='Unpruned')
    # plt.annotate('Unpruned', (100, unpruned_score), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(unpruned_score, (100, unpruned_score), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - unpruned_score

    plt.plot(thresholds, validation_scores, marker='x', label='Val')
    plt.plot(thresholds, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 98.9, 99.7))
    for i, (x, y) in enumerate(zip(thresholds, validation_scores)):
        label = "{:.1f}".format(percentage_pruned[i])

        coords = [
            [0, -15],
            [0, -15],
            [0, -15],
            [-5, -15],
            [0, -15],
            [0, 15],
            [0, -15],
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(thresholds, fully_trained_scores)):
        label = "{:.1f}".format(percentage_pruned[i])
        label_diff = "{:.1f}".format(difference_scores[i])
        coords = [
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [5, 10, -5, -15],
            [5, 10, -5, -15]
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    plt.title("Experiment: Pruning thresholds for mnist limited")
    plt.xlabel("Threshold values.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# COMPLETED PLOTTED
def experiments_threshold_mnist_global_pruning():
    plt.clf()
    # thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999]) * 100
    # validation_scores = np.array([0.9531, 0.9605, 0.9592, 0.9577, 0.9873, 0.9616, 0.9903, 0.9688]) * 100
    # num_heads_pruned = np.array([143, 143, 143, 143, 133, 143, 128, 142])
    # percentage_pruned = (num_heads_pruned/144) * 100
    # fully_trained_scores = np.array([0.9919, 0.9906, 0.9919, 0.9917, 0.994, 0.992, 0.9963, 0.9921]) * 100
    thresholds = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]) * 100
    validation_scores = np.array([0.9531, 0.9605, 0.9592, 0.9577, 0.9873, 0.9616, 0.9903]) * 100
    num_heads_pruned = np.array([143, 143, 143, 143, 133, 143, 128])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.9919, 0.9906, 0.9919, 0.9917, 0.994, 0.992, 0.9963]) * 100
    unpruned_score = 99.40
    plt.plot(100, unpruned_score, marker='x', label='Unpruned')
    # plt.annotate('Unpruned', (100, unpruned_score), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(unpruned_score, (100, unpruned_score), textcoords="offset points", xytext=(0, 10), ha='center')
    difference_scores = fully_trained_scores - unpruned_score

    plt.plot(thresholds, validation_scores, marker='x', label='Val')
    plt.plot(thresholds, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 95, 100))
    for i, (x, y) in enumerate(zip(thresholds, validation_scores)):
        label = "{:.1f}".format(percentage_pruned[i])

        coords = [
            [0, -15],
            [0, -15],
            [0, -15],
            [0, -15],
            [15, -5],
            [15, -5],
            [15, -5],
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(thresholds, fully_trained_scores)):
        label = "{:.1f}".format(percentage_pruned[i])
        label_diff = "{:.1f}".format(difference_scores[i])
        coords = [
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
            [0, 10, 0, -15],
        ]

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][0], coords[i][1]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(coords[i][2], coords[i][3]),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for mnist")
    plt.xlabel("Threshold values.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# COMPLETED PLOTTED
def experiments_threshold_mnist_one_train():
    plt.clf()
    thresholds = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55]) * 100
    validation_scores = np.array([0.096, 0.1857, 0.625, 0.4868, 0.4878, 0.6167]) * 100
    num_heads_pruned = np.array([140, 131, 115, 124, 123, 116])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.9931, 0.9959, 0.996, 0.9954, 0.9930, 0.9961]) * 100
    unpruned_score = 99.40
    plt.plot(100, unpruned_score, marker='x', label='Unpruned')
    # plt.annotate('Unpruned', (100, unpruned_score), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(unpruned_score, (100, unpruned_score), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - unpruned_score

    plt.plot(thresholds, validation_scores, marker='x', label='Val')
    plt.plot(thresholds, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 5, 110))
    for i, (x, y) in enumerate(zip(thresholds, validation_scores)):
        label = "{:.0f}".format(percentage_pruned[i])
        if i == 1:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(-5, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        else:

            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center


    for i, (x, y) in enumerate(zip(thresholds, fully_trained_scores)):
        label = "{:.0f}".format(percentage_pruned[i])
        label_diff = "{:.1f}".format(difference_scores[i])

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        plt.annotate(label_diff,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, -15),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for mnist One Train")
    plt.xlabel("Threshold values.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# COMPLETED PLOTTED
def experiments_threshold_mnist_train_and_prune():
    plt.clf()
    thresholds = np.array([0.94, 0.98, 0.999]) * 100
    num_heads_pruned = np.array([143, 143, 116])
    percentage_pruned = (num_heads_pruned/144) * 100
    fully_trained_scores = np.array([0.99, 0.99, 0.9943]) * 100
    unpruned_score = 99.40
    plt.plot(100, unpruned_score, marker='x', label='Unpruned')
    # plt.annotate('Unpruned', (100, unpruned_score), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(unpruned_score, (100, unpruned_score), textcoords="offset points", xytext=(0, -15), ha='center')
    difference_scores = fully_trained_scores - unpruned_score

    plt.plot(thresholds, fully_trained_scores, marker='x', label='Full')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 98.8, 99.8))
    for i, (x, y) in enumerate(zip(thresholds, fully_trained_scores)):
        label = "{:.1f}".format(percentage_pruned[i])
        label_diff = "{:.1f}".format(difference_scores[i])
        if i == 2:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(-13, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(13, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        else:
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.annotate(label_diff,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, -15),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.title("Experiment: Pruning thresholds for mnist Train and Prune")
    plt.xlabel("Threshold values.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


experiments_threshold_cifar100_prune_and_train()