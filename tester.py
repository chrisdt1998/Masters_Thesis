import numpy as np
import argparse

# head_mask = np.load(r"C:\Users\Gebruiker\Documents\GitHub\Masters_thesis\head_mask_50_cifar100_0.1.npy")
# # head_importance = np.load(r"C:\Users\Gebruiker\Documents\GitHub\Masters_thesis\saved_numpys\head_importance_4.npy")
# # print(head_mask.sum())
# # print(f"num_masked_heads: {12*12 - int(head_mask.sum())}")
# # print(56/14)
# # print(head_importance)
# print(head_mask)
#
# head_mask = np.array(head_mask)
# print((144 - head_mask.sum())/14)
pruned = 0
unpruned = 1
for i in range(0, 13):
    pruned += unpruned * 0.2
    unpruned = unpruned*0.8

    print(pruned, unpruned)
