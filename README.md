# HANDWRITTEN MATHEMATICAL EXPRESSION RECOGNITION

## Introduction

#### Problem
In today's increasingly digitized world, so too does the need for translating handwritings from an image to digital text. The recent boom in deep learning and neural networks has allowed for impressive results in optical character recognition (OCR). This has led to the development of multiple OCR systems for many natural languages like English, Arabic, and Chinese.

However, the language of mathematics presents its own set of challenges for OCR that are still being researched. While some impressive API has been commercialized (Mathpix, MyScript), these systems are closed source and use private datasets for training.

#### Proposal

In this project, I will try to develop a similar system to read, interpret, and translate handwritten mathematical expressions from images to $LaTeX$ code, which is the standard for digital math expression syntax. I will mostly be replicating one of the systems developed in a paper called "Watch, Attend, Parse" (ref. below).

## Model Architecture

My model will take a form an encoder-decoder architecture where the encoder has the job of reading the image and extracting the symbols, which the decoder will take as input and parse into a $LaTeX$ code snippet.

It will look something like this:
![my_model](https://i.imgur.com/fLqA4rp.png)

As shown above, the encoder will take the form of a CNN (VGG, DenseNet) to extract features from the image, and the decoder will be an RNN (LSTM, Transformer) to parse those features. The decoder may or may not be equipped with the attention mechanism depending on the time constraints.

## Dataset

The dataset I will be using is the official [CROHME](https://www.isical.ac.in/~crohme/CROHME_data.html) dataset used as benchmark for most of the research efforts so far.

The CROHME datasets over multiple competitions have been compiled at [this kaggle dataset](https://www.kaggle.com/rtatman/handwritten-mathematical-expressions), which I will choose for ease of use.

Extra datasets for symbol segmentation/classification can be found below:

-    [Handwritten math symbols](https://www.kaggle.com/xainano/handwrittenmathsymbols)
-    [Handwritten math symbols and digits (smaller)](https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset)


## Checklist
-    [x] Cleaning + EDA
-    [x] Preprocessing data
-    [x] CNN encoder
-    [x] Attention mechanism
-    [x] RNN decoder
-    [x] Finishing training
-    [x] Finalizing the pipeline + debugging
-    [x] Streamlit interface and deployment
-    [ ] Presentation slides
-    [ ] DONE :100: (Hope to god I get here)

## Code
-    [Colab Notebook](https://colab.research.google.com/drive/1Frh8sH2iybM7fK733dA5yHG4lYrcCNik?usp=sharing)


## References
1. [Watch, Attend, Parse](http://home.ustc.edu.cn/~xysszjs/paper/PR2017.pdf)
2. [Multi-Scale Attention with Dense Encoder for
Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/1801.03530.pdf)
3. [Offline handwritten mathematical symbol
recognition utilising deep learning](https://arxiv.org/pdf/1910.07395.pdf)
4. [Neural Recognition of Handwritten Mathematical Expressions](https://epub.jku.at/obvulihs/download/pdf/3866590?originalFilename=true)
5. [Improving Attention-Based Handwritten
Mathematical Expression Recognition with Scale
Augmentation and Drop Attention](https://arxiv.org/pdf/2007.10092.pdf)
