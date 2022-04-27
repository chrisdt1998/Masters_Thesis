import matplotlib.pyplot as plt
import numpy as np

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


mask_2_results = [0.8784, 0.8680, 0.8522, 0.8322, 0.8230, 0.7582, 0.6882, 0.4556, 0.3154]
mask_3_results = [0.8782, 0.8484, 0.8368, 0.8116, 0.7142, 0.3514, 0.2026, 0.1616, 0.1264]
mask_2_results = np.array(mask_2_results)
mask_3_results = np.array(mask_3_results)
percentages = np.arange(0, 90, 10)

plt.plot(percentages, mask_2_results, label='mask_2_results')
plt.plot(percentages, mask_3_results, label='mask_3_results')
plt.legend()
plt.title("mask_head_vit_cifar100")
plt.xlabel("Percentage of heads pruned. %")
plt.ylabel("Validation accuracy of the model. %")
plt.show()