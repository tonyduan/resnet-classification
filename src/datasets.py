import torch
import random
from torchvision import datasets, transforms
from src.lib.zipdata import ZipData


class PrecisionTransform(object):
    
    def __init__(self, precision):
        self.precision = precision

    def __call__(self, x):
        if self.precision == "float":
            return x.float()
        if self.precision == "half":
            return x.half()
        if self.precision == "double":
            return x.double()


def get_dim(name):
    if name.startswith("cifar"):
        return 3 * 32 * 32
    if name == "svhn":
        return 3 * 32 * 32
    if name == "mnist":
        return 28 * 28
    if name == "fashion":
        return 28 * 28
    if name == "imagenet":
        return 3 * 224 * 224

def get_num_labels(name):
    if name == "imagenet":
        return 1000
    if name == "cifar100":
        return 100
    return 10

def get_label_names(name):
    if name == "cifar":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if name == "cifar100":
        return ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates", "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp", "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor"]
    else:
        return [str(i) for i in range(1, get_num_labels(imagenet))]
    raise ValueError

def get_normalization_shape(name):
    if name.startswith("cifar"):
        return (3, 1, 1)
    if name == "imagenet":
        return (3, 1, 1)
    if name == "svhn":
        return (3, 1, 1)
    if name == "mnist":
        return (1, 1, 1)
    if name == "fashion":
        return (1, 1, 1)

def get_normalization_stats(name):
    if name == "cifar" or name == "cifar100":
        return {"mu": [0.4914, 0.4822, 0.4465], "sigma": [0.2023, 0.1994, 0.2010]}
    if name == "imagenet":
        return {"mu": [0.485, 0.456, 0.406], "sigma": [0.229, 0.224, 0.225]}
    if name == "svhn":
        return {"mu": [0.436, 0.442, 0.471], "sigma": [0.197, 0.200, 0.196]}
    if name == "mnist":
        return {"mu": [0.1307,], "sigma": [0.3081,]}
    if name == "fashion":
        return {"mu": [0.2849,], "sigma": [0.3516,]}

def get_dataset(name, split, precision):

    precision_transform = PrecisionTransform(precision)

    if name == "cifar" and split == "train":
        return datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),
                                                              precision_transform]))

    if name == "cifar" and split == "test":
        return datasets.CIFAR10("./data/cifar_10", train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              precision_transform]))

    if name == "cifar100" and split == "train":
        return datasets.CIFAR100("./data/cifar_100", train=True, download=True,
                                 transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               precision_transform]))

    if name == "cifar100" and split == "test":
        return datasets.CIFAR100("./data/cifar_100", train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               precision_transform]))

    if name == "imagenet" and split == "train":
        return ZipData("/mnt/imagenet/train.zip", "/mnt/imagenet/train_map.txt",
                       transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           precision_transform]))

    if name == "imagenet" and split == "test":
        return ZipData("/mnt/imagenet/val.zip", "/mnt/imagenet/val_map.txt", 
                       transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           precision_transform]))

    if name == "mnist":
        return datasets.MNIST("./data/mnist", train=(split == "train"), download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            precision_transform]))
    if name == "fashion":
        return datasets.FashionMNIST("./data/fashion", train=(split == "train"), download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   precision_transform]))

    if name == "svhn":
        return datasets.SVHN("./data/svhn", split=split, download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           precision_transform]))

    raise ValueError

