import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

IMG_SHAPE = (300,484,1)

def decode_img(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_ANYCOLOR)
    return img


def binarize(image, uploaded=False):
    if uploaded:
        img = cv2.adaptiveThreshold(image,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    5, 3)
    else:
        _,img = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    return img


def tensorize(image):
    img = tf.cast(image, 'float32')
    if len(img.shape) == 2:
        img = tf.expand_dims(img, axis=-1)
    img = tf.image.resize(img, [IMG_SHAPE[0], IMG_SHAPE[1]])
    img = img/255
    return img

def preprocess_st(decoded_image, uploaded=False):
    img = decoded_image

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGBA2GRAY)

    img = binarize(img, uploaded=uploaded)
    img = tensorize(img)
    return img
