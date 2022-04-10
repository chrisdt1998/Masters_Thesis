from transformers import ViTFeatureExtractor, ViTModel
import torch
import torchvision

# mnist_train = torchvision.datasets.MNIST('/Users/chris/Documents/GitHub/Masters_Thesis/mnist_dataset',
#                                         download=True,
#                                         train=True)
#
# data_loader_train = torch.utils.data.DataLoader(mnist_train,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=args.nThreads)
#
# mnist_test = torchvision.datasets.MNIST('/Users/chris/Documents/GitHub/Masters_Thesis/mnist_dataset',
#                                         download=True,
#                                         train=False)
#
# data_loader_test = torch.utils.data.DataLoader(mnist_test,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=args.nThreads)

from transformers import ViTModel, ViTConfig

# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()

# Initializing a model from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)

# Accessing the model configuration
configuration = model.config
print(configuration)

