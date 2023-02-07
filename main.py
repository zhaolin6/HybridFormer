# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep model (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import cv2
import torch
import torch.utils.data as data
from torchsummary import summary
import openpyxl

# Numpy, scipy, scikit-image, spectral
import pandas as pd
import numpy as np
import sklearn.svm
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
from skimage import io
# Visualization
import seaborn as sns
import visdom
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device, get_train
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG, HyperX4Test
from models import get_model, train, test, save_model

import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='Salinas', choices=dataset_names,
                    help="Dataset to use: IndianPines; PaviaU; Salinas")
parser.add_argument('--model', type=str, default=["hybridformer"])
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')

# group_dataset.add_argument('--training_sample', type=float, default=[5],
#                     help="Percentage of samples to use for training (default: 10%)")
# group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
#                     " (random sampling or disjoint, default: random)",
#                     default='fixed')

group_dataset.add_argument('--training_sample', type=float, default=[0.01],
                    help="Percentage of samples to use for training (default: 10%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
"""
    sampling_mode 表示取样模式，random表示随机取样, fixed 表示固定数目取样, disjoint不太清楚 
"""

group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int, default=15,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',  default=True,
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')

group_da.add_argument('--flip_augmentation', action='store_true', default = True,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default = True,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default = True,
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGES = args.training_sample

# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
# DATASET = args.dataset
# Model name
MODELS = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env="ALL")
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")



datasets = ['HyRANK']  # 'Houston', 'WHU-HI', 'Chikusei', 'HyRANK'
patch_sizes = [[3, 5]]
dataset_result = []
repeat_term = 1
d_h = [[2, 4]]

sheet_num = 1
# for dh in d_h:
#     MODEL = MODELS[0]
for MODEL in MODELS:
    dh = d_h[0]

    args.MODEL = MODEL
    for patch_size in patch_sizes:
        for SAMPLE_PERCENTAGE in SAMPLE_PERCENTAGES:
            for DATASET in datasets:
                args.dataset = DATASET
                mean_oa = 0
                mean_aa = 0
                mean_kappa = 0
                num_c = 0
                args.DATASET = DATASET
                for j in range(repeat_term):
                    hyperparams = vars(args)
                    # Load the dataset
                    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(dataset_name=DATASET, target_folder=FOLDER, patch_size=PATCH_SIZE)
                    # Number of classes
                    # N_CLASSES = len(LABEL_VALUES) -  len(IGNORED_LABELS)
                    N_CLASSES = len(LABEL_VALUES)
                    # Number of bands (last dimension of the image tensor)
                    N_BANDS = img.shape[-1]

                    num_c = num_c + len(LABEL_VALUES)

                    # Parameters for the SVM grid search
                    SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                                           'C': [1, 10, 100, 1000]},
                                       ]

                    if palette is None:
                        # Generate color palette
                        palette = {0: (0, 0, 0)}
                        for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
                            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
                    invert_palette = {v: k for k, v in palette.items()}

                    def convert_to_color(x):
                        return convert_to_color_(x, palette=palette)

                    def convert_from_color(x):
                        return convert_from_color_(x, palette=invert_palette)

                    # Instantiate the experiment based on predefined networks
                    hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
                    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

                    # Show the image and the ground truth
                    # display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)

                    # color_gt = convert_to_color(gt)
                    # display_dataset(img, color_gt, RGB_BANDS, LABEL_VALUES, palette, viz)

                    if DATAVIZ:
                        # Data exploration : compute and show the mean spectrums
                        mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                                           ignored_labels=IGNORED_LABELS)
                        plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

                    results = []
                    # run the experiment several times
                    for run in range(N_RUNS):
                        if TRAIN_GT is not None and TEST_GT is not None:
                            train_gt = open_file(TRAIN_GT)
                            test_gt = open_file(TEST_GT)
                        elif TRAIN_GT is not None:
                            train_gt = open_file(TRAIN_GT)
                            test_gt = np.copy(gt)
                            w, h = test_gt.shape
                            test_gt[(train_gt > 0)[:w,:h]] = 0
                        elif TEST_GT is not None:
                            test_gt = open_file(TEST_GT)
                        else:
                            # Sample random training spectra
                            train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, ignore=IGNORED_LABELS, mode=SAMPLING_MODE,)

                        if MODEL == 'SVM_grid':
                            print("Running a grid search SVM")
                            # Grid search SVM (linear and RBF)
                            img = img[PATCH_SIZE // 2:-(PATCH_SIZE // 2), PATCH_SIZE // 2:-(PATCH_SIZE // 2), :]
                            X_train, y_train = build_dataset(img, train_gt,
                                                             ignored_labels=IGNORED_LABELS)
                            class_weight = 'balanced' if CLASS_BALANCING else None
                            clf = sklearn.svm.SVC(class_weight=class_weight)
                            clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
                            clf.fit(X_train, y_train)
                            print("SVM best parameters : {}".format(clf.best_params_))
                            prediction = clf.predict(img.reshape(-1, N_BANDS))
                            save_model(clf, MODEL, DATASET)
                            prediction = prediction.reshape(img.shape[:2])
                        elif MODEL == 'SVM':
                            X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
                            class_weight = 'balanced' if CLASS_BALANCING else None
                            clf = sklearn.svm.SVC(class_weight=class_weight)
                            clf.fit(X_train, y_train)
                            save_model(clf, MODEL, DATASET)
                            prediction = clf.predict(img.reshape(-1, N_BANDS))
                            prediction = prediction.reshape(img.shape[:2])
                        elif MODEL == 'SGD':
                            X_train, y_train = build_dataset(img, train_gt,
                                                             ignored_labels=IGNORED_LABELS)
                            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
                            scaler = sklearn.preprocessing.StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            class_weight = 'balanced' if CLASS_BALANCING else None
                            clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3, average=10)
                            clf.fit(X_train, y_train)
                            save_model(clf, MODEL, DATASET)
                            prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
                            prediction = prediction.reshape(img.shape[:2])
                        elif MODEL == 'nearest':
                            img = img[PATCH_SIZE // 2:-(PATCH_SIZE // 2), PATCH_SIZE // 2:-(PATCH_SIZE // 2), :]
                            X_train, y_train = build_dataset(img, train_gt,
                                                             ignored_labels=IGNORED_LABELS)
                            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
                            class_weight = 'balanced' if CLASS_BALANCING else None
                            clf = KNeighborsClassifier(weights='distance')
                            clf = sklearn.model_selection.GridSearchCV(clf, {'n_neighbors': [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4)
                            clf.fit(X_train, y_train)
                            save_model(clf, MODEL, DATASET)
                            prediction = clf.predict(img.reshape(-1, N_BANDS))
                            prediction = prediction.reshape(img.shape[:2])
                        else:
                            if CLASS_BALANCING:
                                weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
                                weights = torch.from_numpy(weights)
                                weights = weights.to(CUDA_DEVICE)
                                hyperparams['weights'] = weights.float()
                            # Neural network
                            model, optimizer, loss, hyperparams = get_model(MODEL, ps=patch_size, h_d=dh, **hyperparams)

                            # Generate the dataset
                            train_dataset = HyperX(img, train_gt, **hyperparams)
                            test_dataset = HyperX4Test(img, test_gt, **hyperparams)

                            train_loader = data.DataLoader(train_dataset,
                                                           batch_size=hyperparams['batch_size'],
                                                           pin_memory=hyperparams['device'],
                                                           shuffle=True)

                            print(hyperparams)
                            print("Network :")

                            try:
                                train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                                      scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                                      supervision=hyperparams['supervision'], val_loader=None,
                                      display=viz, args=args)  # display = viz
                            except KeyboardInterrupt:
                                pass

                            probabilities = test(model, img, hyperparams, args=args)
                            prediction = np.argmax(probabilities, axis=-1)
                            prediction = prediction[PATCH_SIZE // 2:-(PATCH_SIZE // 2), PATCH_SIZE // 2 : -(PATCH_SIZE // 2)]

                        run_results = metrics(prediction, test_gt, ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)
                        mask = np.zeros(gt.shape, dtype='bool')
                        for l in IGNORED_LABELS:
                            mask[gt == l] = True
                        prediction[mask] = 0

                        color_prediction = convert_to_color(prediction)
                        display_predictions(color_prediction, viz, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")

                        im = Image.fromarray(color_prediction)
                        im_name = "{}_{}_{}.jpg".format(MODEL, DATASET, repeat_term)
                        im.save(os.path.join('images', im_name))
                        results.append(run_results)
                        print(results[0])

