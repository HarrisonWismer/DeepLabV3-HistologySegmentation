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
There are 4 executables total in this repositoty:
1. train.py -> Used to train a model from scratch.
2. predict_whole_image.py -> Used to create a prediction mask given a whole slide image.
3. visualize_results.py -> Used to overlay the prediction mask on the original image in an interactive napari viewer.
4. get_qc_metrics.py -> Used to get accuracy values for a prediction mask compared to a ground-truth label mask.

## 1. train.py
This is likely the first script you will run, as it trains and saves a DeepLabV3+ model to be later used for prediction tasks. To run this script, after cloning this repository make sure to copy all of your training images into the training_data/imgs folder and all of you training
masks into the training_data/masks folder. If these folders are empty, the model will not train.

#### Training Data Generation From H&E Images:
In order to avoid the generally large sizes of H&E images, and to provide the model with enough example images for training, a tiling approach was used to
create many training tiles to train the model with. This was done in QuPath where whole-slide annotations were present and overlaid over their corresponding regions
in the original image. A script was used to tile the entire image using a specified tile size. Pairs of tiles consisting of the original image tile and its corresponding mask were saved and placed in the "imgs" and "masks" folders for training. It is a good idea to remember the tile size used during this process, as we will need it later to compute downsampling values during prediction.

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

## 2. predict_whole_image.py
Since the goal of the project these scripts were designed for was to train on H&E data, the predict_whole_image.py script was designed to
read in a WHOLE SLIDE IMAGE. Giving the model an entire image to predict is not feasible in this case because of the tiling approach used during training.
To handle this, another tiling approach is used in which the input image for prediction is tiled, each tile is predicted and then stitched back together
to create the final image and it's corresponding prediction overlay.

The positional (required, in the order specified) arguments for predict_whole_image.py are:
1. model_path -> The path to the model trained using train.py.
2. image_path -> The path to the whole-slide image that will be predicted.
3. tile_size -> The size of each tile specified when doing the INITIAL TILING (Essentially, the size of the training images). This is not necessarily the same as image_size from train.py.
4. num_classes -> The number of classes the model will predict (should be the same as when training).

The only optional argument for predict_whole_image.py is:
1. show_viewer -> A boolean specifying whether or not to open a napari viewer for interactive viewing of the predictions. This viewing can also be done later in visualize_results.py.

After running the script, the following should be saved to the "predictions" folder in the cloned repository:
1. A (potentially) downsampled version of the original image.
2. A prediction mask/overlay containing the predictions for the whole image.

## 3. visualize_results.py
Using the output of prediction_whole_image.py, a napari viewer session can be opened where the prediction mask is overlaid upon the original image.
This viewer is interactive, and the purpose of this script is to be able to zoom in and out to get a sense of how good the predictions are.

The only two inputs to this script are:
1. image_path -> The path to the saved raw image from predict_whole_image.py.
2. overlay_path -> The path to the saved prediction mask from predict_whole_image.py.


## 4. get_qc_metrics.py
Using the prediction mask generated by the previously described scripts, the accuracy of the predictions can be calculated if a ground-truth mask is supplied.

Inputs:
1. truth_path -> The path to a label-image with ground truth values.
2. prediction_path -> The path to a label-image with prediction values.
3. num_classes -> The number of classes present.
4. class_names, -c (Optional) -> The names of the classes present. If not specified, the script will output numerical categories (ie. 1,2,3,...) as class names. The number of class names specified should be equal to (num_classes - 1) since the background "class" does not need to be named.

Outputs:
1. A global accuracy value.
2. A "class" accuracy value for each class.
