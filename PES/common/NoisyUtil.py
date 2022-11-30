import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

import numpy as np
from math import inf
from scipy import stats
import PIL.Image as Image
from numpy.testing import assert_array_almost_equal
import pdb

def dataset_split(train_images, train_labels, noise_rate=0.5, noise_type='symmetric', split_per=0.9, random_seed=1, num_classes=10, include_noise=False):

    if include_noise:
        noise_rate = noise_rate * (1 - 1 / num_classes)
        print("include_noise True, new real nosie rate:", noise_rate)

    clean_train_labels = train_labels[:, np.newaxis]
    if(noise_type == 'pairflip'):
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels, noise=noise_rate,
                                                                            random_state=random_seed, nb_classes=num_classes)

    elif(noise_type == 'instance'):
        norm_std = 0.1
        if(len(train_images.shape) == 2):
            feature_size = train_images.shape[1]
        else:
            feature_size = 1
            for i in range(1, len(train_images.shape)):
                feature_size = int(feature_size * train_images.shape[i])

        if torch.is_tensor(train_images) is False:
            data = torch.from_numpy(train_images)
        else:
            data = train_images

        data = data.type(torch.FloatTensor)
        targets = torch.from_numpy(train_labels)
        dataset = zip(data, targets)
        noisy_labels = get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, random_seed)
    elif(noise_type == 'oneflip'):
        noisy_labels, real_noise_rate, transition_matrix = noisify_oneflip(clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    else:
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels, noise=noise_rate,
                                                                                        random_state=random_seed, nb_classes=num_classes)

    clean_train_labels = clean_train_labels.squeeze()
    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]
    train_clean_labels, val_clean_labels = clean_train_labels[train_set_index], clean_train_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels, train_clean_labels, val_clean_labels


def get_instance_noisy_label(n, newdataset, labels, num_classes, feature_size, norm_std, seed):
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)
    if torch.cuda.is_available():
        W = torch.FloatTensor(W).cuda()
    else:
        W = torch.FloatTensor(W)
    for i, (x, y) in enumerate(newdataset):
        if torch.cuda.is_available():
            x = x.cuda()
            x = x.reshape(feature_size)

        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l1 = [i for i in range(label_num)]
    new_label = [np.random.choice(l1, p=P[i]) for i in range(labels.shape[0])]

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    return np.array(new_label)


def noisify_oneflip(y_train, noise, random_state=1, nb_classes=10):
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[1, 1], P[1, 2] = 1. - n, n
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


# basic function
def multiclass_noisify(y, P, random_state=1):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # i is np.array, such as [1]
        if not isinstance(i, np.ndarray):
            i = [i]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


class Train_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


class Semi_Labeled_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return out1, out2, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


class Semi_Unlabeled_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = self.train_data.shape[0]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        return out1, out2, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data


def getNoisyData(seed, dataset, data_root, data_percent, noise_type, noise_rate, include_noise=False):
    """
    return train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, val_clean_labels
    """
    if dataset == "CIFAR10" or dataset == "cifar10":
        num_classes = 10
        train_set = CIFAR10(root=data_root, train=True, download=False)
    elif dataset == "CIFAR100" or dataset == "cifar100":
        num_classes = 100
        train_set = CIFAR100(root=data_root, train=True, download=False)

    return dataset_split(train_set.data, np.array(train_set.targets), noise_rate, noise_type, data_percent, seed, num_classes, include_noise)


def getDataLoaders(seed, dataset, data_root, data_percent, noise_type, noise_rate, batch_size, is_clean=False, no_aug=False):

    if not isinstance(noise_rate, float):
        raise ValueError("noise_rate is not float")

    if dataset == "CIFAR10" or dataset == "cifar10":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        num_classes = 10
        train_set = CIFAR10(root=data_root, train=True, download=False)
        test_set = CIFAR10(root=data_root, train=False, transform=transform_test, download=True)
    elif dataset == "CIFAR100" or dataset == "cifar100":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        num_classes = 100
        train_set = CIFAR100(root=data_root, train=True, download=False)
        test_set = CIFAR100(root=data_root, train=False, transform=transform_test, download=True)

    train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, val_clean_labels = dataset_split(train_set.data, np.array(train_set.targets), noise_rate, noise_type, data_percent, seed, num_classes)

    if no_aug:
        transform_train = transform_test

    if is_clean:
        print("train with clean labels")
        train_dataset = Train_Dataset(train_data, train_clean_labels, transform_train)
        val_dataset = Train_Dataset(val_data, val_clean_labels, transform_test)
    else:
        train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
        val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_transition_matrix(dataset, noise_type, noise_rate):
    if dataset == "CIFAR10" or dataset == "cifar10":
        nb_classes = 10
    elif dataset == "CIFAR100" or dataset == "cifar100":
        nb_classes = 100

    if noise_type == "symmetric":
        transition_matrix = np.ones((nb_classes, nb_classes))
        transition_matrix = (noise_rate / (nb_classes - 1)) * transition_matrix
        if noise_rate > 0.0:
            transition_matrix[0, 0] = 1. - noise_rate
            for i in range(1, nb_classes - 1):
                transition_matrix[i, i] = 1. - noise_rate
                transition_matrix[nb_classes - 1, nb_classes - 1] = 1. - noise_rate

    elif noise_type == "pairflip":
        transition_matrix = np.eye(nb_classes)
        if noise_rate > 0.0:
            transition_matrix[0, 0], transition_matrix[0, 1] = 1. - noise_rate, noise_rate
            for i in range(1, nb_classes - 1):
                transition_matrix[i, i], transition_matrix[i, i + 1] = 1. - noise_rate, noise_rate
            transition_matrix[nb_classes - 1, nb_classes - 1], transition_matrix[nb_classes - 1, 0] = 1. - noise_rate, noise_rate

    return transition_matrix
