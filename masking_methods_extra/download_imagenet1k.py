# import opendatasets as od

# od.download('https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz')

from datasets import load_dataset

dataset = load_dataset("imagenet-1k", use_auth_token=True)