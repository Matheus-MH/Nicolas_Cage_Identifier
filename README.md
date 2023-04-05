# Nicolas_Cage_Identifier
Nicolas Cage image identifier developed using Convolutional Neural Network with Tensorflow on Python.

## Execution
This program wad made in Pycharm using the following libraries:

- from tkinter import filedialog
- import customtkinter
- import numpy as np
- import os
- import cv2 as cv
- import tensorflow as tf
- from tensorflow.keras import models, layers
= import matplotlib.pyplot as plt

in order to run it you must install them in your IDE.

This Model already have a trained file, you can run it directly without training another set.

Since GitHub has some restriction on upload arquives, i had to zip the images used to train the model.

In order to make them work again, unzip the files on Nicolas Cage and People folders, delete the ZIP files and leave only the images on the respective files.

## Training Images

This program can be used to train whatever set of images you like.

To train another set, delete the images on Nicolas Cage and People folders and add the ones you want to train.

This program separate them in two diferent kinds based on the two folders of images you placed on Image directory.

If you'd like to train another set, place the images on the directory, click on the Train button in the GUI and wait for it to fully train over your new set, this operation might take a while to complete.
