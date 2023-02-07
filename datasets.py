# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
import cv2
import h5py
import scipy.io as io

from tqdm import tqdm
from sklearn import preprocessing
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
        'PaviaC': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                     'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'Houston': {
            'urls': ['',
                     ''],
            'img': 'Houston.mat',
            'gt': 'Houston_gt.mat'
            },
        'PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                     'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Botswana': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                     'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            'img': 'Botswana.mat',
            'gt': 'Botswana_gt.mat',
            },
        'Salinas': {
            'img': 'Salinas.mat',
            'gt': 'Salinas_gt.mat',
        },
        'HyRANK': {
            'img': 'Dioni.mat',
            'gt': 'Dioni_GT.mat',
        },
        'WHU-HI': {
             'img': 'WHU-HI-HONGHU.tif',
             'gt': 'WHU-HI-HONGHU_gt.tif',
         },
        'Chikusei': {
             'img': 'HyperspecVNIR_Chikusei_20140729.mat',
             'gt': 'HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat',
         },
    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG, patch_size = 11, model='train'):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = []
    elif dataset_name == 'HyRANK':
        # Load the image
        img = open_file(folder + 'Dioni.mat')['Dioni']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'Dioni_GT.mat')['Dioni_GT']

        label_values = ['0', '1',  '2', '3', '4', '5', '7', '9', '10', '11', '12', '13', '14']
        ignored_labels = [0]
        tmp = (gt > 8)
        gt[tmp] -= 1
        tmp = (gt > 6)
        gt[tmp] -= 1
        u = np.unique(gt)
        a = 1
    #
    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = []
        # ignored_labels = [0, 1, 4, 6, 7, 9, 13, 15, 16]
        # ignored_labels = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']
        rgb_bands = (75, 33, 15)
        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]
        ignored_labels = []
    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        ignored_labels = []

    elif dataset_name == 'Salinas':
        # Load the image
        img = open_file(folder + 'Salinas.mat')
        img = img['salinas_corrected']
        # img = open_file(folder + 'Salinas_PCA_30.mat')
        # img = img['q']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']

        label_values = ["Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
                        "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery",
                        "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
                        "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
                        "Vinyard_untrained", "Vinyard_vertical_trellis"]

        ignored_labels = []
    elif dataset_name == 'Houston':
        img = open_file(folder + 'Houston.mat')
        img = img['Houston']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']
        label_values = ["Undefined", "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water",
                        "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2"
            , "Tennis Court", "Running Track"]

        ignored_labels = []
    elif dataset_name == 'WHU-HI':
        ppp = r'Datasets'
        folder = os.path.join(ppp, "WHU-HI-HONGHU")
        img = open_file(os.path.join(folder, "WHU-Hi-HongHu.tif")).transpose(1, 2, 0)
        rgb_bands = (55, 41, 12)
        gt = open_file(os.path.join(folder, "WHU-Hi-HongHu_gt.tif"))

        # label_values = ["Undefined", "Strawberry", "Cowpea",
        #                 "Soybean","Sorghum",
        #                 "Water spinach", "Watermelon", "Greens",
        #                 "Trees", "Grass", "Red roof",
        #                 "Gray roof", "Plastic","Bare soil","Road",
        #                 "Bright object", "Water"]
        ### HongHu
        label_values = ["Undefined", "Red roof	", "Road", "Bare soil",
                        "Cotton", "Cotton firewood", "Rape", "Chinese cabbage",
                        "Pakchoi", "Cabbage", "Tuber mustard",
                        "Brassica parachinensis", "Brassica chinensis", "Small Brassica chinensis",
                        "Lactuca sativa", "Celtuce", "Film covered lettuce", "Romaine lettuce",
                        "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree"]
        ignored_labels = []
    elif dataset_name == 'Chikusei':
        # folder = os.path.join(folder, "Chikusei")
        rgb_bands = (55, 41, 12)
        img = h5py.File(os.path.join(folder, "HyperspecVNIR_Chikusei_20140729.mat"), 'r')['chikusei'][:].transpose(2, 1, 0)  # 128, 2335, 2517
        gt, _, _ = io.loadmat(os.path.join(folder, "HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat"))['GT'][0, 0]
        label_values = ["Undefined", "Water	","Bare soil(school)","Bare soil(park)",
                        "Bare soil(farmland)","Natural plants","Weeds in farmland",
                        "Forest", "Grass", "Rice field(grown)", "Rice field(first stage)",
                        "Row crops", "Plastic house", "Manmade(non-dark)",
                        "Manmade(dark)", "Manmade(blue)","Manmade(red)","Manmade grass",
                        "Asphalt", "Paved ground"]
        ignored_labels = []
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0

    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))

    data = preprocessing.scale(data)
    # data  = preprocessing.minmax_scale(data)

    img = data.reshape(img.shape)

    img = cv2.copyMakeBorder(img, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2,
                             cv2.BORDER_REFLECT)

    #######################################
    # 加个padding=
    # img = np.pad(img, [(5, 5), (5, 5), (0, 0)], mode = "constant")
    #######################################

    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        mask = None
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)

        p = self.patch_size // 2

        #######################################
        # + 了 padding
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        #######################################

        # self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]

        self.label = np.pad(self.label, [(self.patch_size // 2, self.patch_size // 2), (self.patch_size // 2, self.patch_size // 2)],
               mode="constant")

        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        # alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=0.05, size=data.shape)
        return data + noise
        # data2 = np.zeros_like(data)
        # for  idx, value in np.ndenumerate(label):
        #     if value not in self.ignored_labels:
        #         l_indices = np.nonzero(self.labels == value)[0]
        #         l_indice = np.random.choice(l_indices)
        #         assert(self.labels[l_indice] == value)
        #         x, y = self.indices[l_indice]
        #         data2[idx] = self.data[x,y]
        # return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]

        #######################################
        label = self.label
        x2, y2 = x + self.patch_size, y + self.patch_size
        data = self.data[x:x2, y:y2]
        label = label[x:x2, y:y2]

        # # if self.flip_augmentation and self.patch_size > 1:
        # if self.flip_augmentation:
        #     data, label = self.flip(data, label)
        #
        # # if self.radiation_augmentation and np.random.random() < 0.1:
        # if self.radiation_augmentation:
        #     data = self.radiation_noise(data)
        #
        # # if self.mixture_augmentation and np.random.random() < 0.2:
        # if self.mixture_augmentation:
        #     data = self.mixture_noise(data, label)

        if self.flip_augmentation and np.random.random() < 0.5:
            data, label = self.flip(data, label)

        if self.radiation_augmentation and np.random.random() < 0.5:
        # if self.radiation_augmentation:
            data = self.radiation_noise(data)

        if self.mixture_augmentation and np.random.random() < 0.5:
        # if self.mixture_augmentation:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:  # True
            label = label[self.patch_size // 2, self.patch_size // 2]

        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        return data, label


class HyperX4Test(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX4Test, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)

        p = self.patch_size // 2

        #######################################
        # + 了 padding
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        #######################################

        # self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]

        self.label = np.pad(self.label, [(self.patch_size // 2, self.patch_size // 2), (self.patch_size // 2, self.patch_size // 2)],
               mode="constant")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]

        #######################################
        label = self.label
        x2, y2 = x + self.patch_size, y + self.patch_size
        data = self.data[x:x2, y:y2]
        label = label[x:x2, y:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:  # True
            label = label[self.patch_size // 2, self.patch_size // 2]

        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        return data, label
