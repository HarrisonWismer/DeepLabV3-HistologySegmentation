# DeepLabV3 - H&E Segmentation

## Resources:
These scripts implements the DeepLabV3+ model for multi-class semantic segmentation as outlined in:

[Chen, LC., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science(), vol 11211. Springer, Cham](https://doi.org/10.1007/978-3-030-01234-2_49)

Large amounts of code for data loading and model architecture creation were taken and either used directly or were slightly adapted from the keras examples page:

[Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus/)

## Overview
As part of a larger project that hoped to achieve sufficient segmentation and the ability to classify various structures found in H&E pathology images, the DeepLabV3+ model
was utilized. The purpose of these scripts is to create a simple, streamlined process from which others can train a model for their own purposes. Though the focus of this project is solely on H&E images, the pre-trained weights from ImageNet that are used in this model would likely lend themselves to many other kinds of multi-class segmentation.

## Prerequisites
Found in the "environments" folder within this repository are two conda environment creation .yml files. One of the files installs all the necessary packages but excludes tensorflow, while the other file installs tensorflow in accordance with the instructions for the WINDOWS install. If the environment with tensorflow included does not work, use the other environment file and follow the [Tensorflow installation instructions for your machine here](https://www.tensorflow.org/install).

## Running Overview
There are only 3 executables total in this repositoty:
    1. train.py -> Used to train a model from scratch.
    2. predict_whole_image.py -> Used to create a prediction mask given a whole slide image.
    3. visualize_results.py -> Used to overlay the prediction mask on the original image in an interactive napari viewer.

## 1. train.py
This is likely the first script you will run, as it trains and saves a DeepLabV3+ model to be later used for prediction tasks.


To run this script, after cloning this repository make sure to copy all of your training images into the training_data/imgs folder and all of you training
masks into the training_data/masks folder. If these folders are empty, the model will not train.

The optional parameters for train.py are:
    * image_size -i -> The input size for each image for the model. For an integer n specified, each image will be scaled to (n x n).
    * num_classes -n -> An integer specifying the number of classes the model will be trained to predict.
    * val_split -v -> A float f (0.0 <= f <= 1.0) specifying the proportion of the training data to be used for validation.
    * batch_size -b -> The batch size to be used during training.
    * learning_rate -l -> The learning rate to be used during training.
    * numEpochs -e -> The number of epochs to train for.
    * save_path -> The name to be used when saving the model. The model is saved into the "models" folder of the cloned repository.

After running the script, a saved model will be found in the "models" folder of the cloned repository to then be used in prediction tasks.
Additionally, the final training and validation losses and accuracies will be printed to STDOUT.



 
