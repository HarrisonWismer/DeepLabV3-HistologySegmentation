# DeepLabV3 - H&E Segmentation

## Resources:
These scripts implement the DeepLabV3+ model for multi-class semantic segmentation as outlined in:

[Chen, LC., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science(), vol 11211. Springer, Cham](https://doi.org/10.1007/978-3-030-01234-2_49)

Code for data loading and model architecture creation were taken and either used directly or were slightly adapted from the keras examples page:

[Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus/)

## Overview
The DeepLabV3+ model was utilized as part of a lrager project aiming to predict various structures found in histological staining images. The purpose of these scripts is to create a simple, streamlined process where, as long as they have a set of raw images as well as a set of corresponding label masks, anyone can train a model to perform multi-class semantic segmentation for their own purposes. Though the focus of this project is to be able to predict larger structures within H&E images, the pre-trained weights from ImageNet built into the model architecture would likely lend themselves to other multi-class segmentation tasks.

## Prerequisites
Found in the "environments" folder within this repository are two conda environment creation .yml files. One of the files installs all the necessary packages but excludes tensorflow, while the other file installs tensorflow in accordance with the instructions for the WINDOWS install. If the environment with tensorflow included does not work, use the other environment file and follow the [Tensorflow installation instructions for your machine here](https://www.tensorflow.org/install).

## Running Overview
There are 4 executables in total in this repositoty:
1. train.py -> Used to train a model from scratch given a set of raw images and their corresponding ground-truth label masks.
2. predict_whole_image.py -> Used to create a prediction mask given a whole slide image via a tiling procedure.
3. visualize_results.py -> Used to overlay the prediction mask on the original image in an interactive napari viewer.
4. get_qc_metrics.py -> Used to get F1-Score for each class when given a prediction label mask and a ground-truth label mask.

## 1. train.py
This is likely the first script that will be run, as it trains and saves a DeepLabV3+ model to be later used for prediction tasks. After cloning this repository, make sure to copy all of your training images into the training_data/imgs folder and all of your training
masks into the training_data/masks folder. If these folders are empty, the model will not train on your images.

##### Input:
* Training images and their corresponding ground-truth masks must be placed in the training_data/imgs and training_data/masks respectively.

The optional parameters for train.py are:
* image_size -i -> The input size for each image for the model. For an integer n specified, each image will be scaled to (n x n). Default 256.
* num_classes -n -> An integer specifying the number of classes the model will be trained to predict. Default 4.
* val_split -v -> A float f (0.0 <= f <= 1.0) specifying the proportion of the training data to be used for validation. Default .2.
* batch_size -b -> The batch size to be used during training. default 8.
* learning_rate -l -> The learning rate to be used during training. Default .001.
* numEpochs -e -> The number of epochs to train for. Default 3.
* save_path -> The name to be used when saving the model. The model is saved into the "models" folder of the cloned repository.

##### Output:
* After running the script, a saved model will be found in the "models" folder within the cloned repository.
* model_info.txt -> A text file with final loss and accuracy values for both the training data and validation data. Also includes the parameters used to train the model.

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
Using the prediction mask generated by the previously described scripts, the quality control metrics of the predictions can be calculated if a ground-truth mask is supplied.

Inputs:
1. truth_path -> The path to a label-image with ground truth values.
2. prediction_path -> The path to a label-image with prediction values.
3. num_classes -> The number of classes present.
4. class_names, -c (Optional) -> The names of the classes present. If not specified, the script will output numerical categories (ie. 0,1,2,3,...) as class names.

Output:
1. A F1 Score for each class

# Whole-Slide Multiplexed Immunofluorescence and H&E Segmentation Workflow

## Background:
Light microscopy represents one of the 3 main diagnostic modalities for the pathologic interpretation of kidney biopsies. Routine pathology evaluation of kidney biopsies is based on pattern recognition combined with quantitative and/or semi-quantitative assessment of the abnormalities in the four main compartments of the kidney, i.e., glomeruli, tubules, interstitium, and vessels. Computer-assisted interpretation of the biopsies on whole slide digital images has the potential to improve the quality of the pathology reports via standardization and precise quantitative read-outs. Segmentation of biopsies into the main compartments is the prerequisite for computer-assisted interpretation of kidney biopsies. Here, we will develop computational algorithms for compartmental segmentation of the kidney tissue. 

## Objective:
To develop compartmental segmentation algorithms for the kidney on H&E slides via supervised machine learning using multiplexed IF images and the corresponding H&E slides on normal kidney biopsies; compartments (objects) to be segmented are as follows:

1. Glomeruli
    1. Mesangium
    2. Capillaries
    3. Podocytes

2. Tubules
    1. Proximal
    2. Distal
    3. Collecting Ducts

3. Interstitium

4. Vessels
    1. Capillaries
    2. Arteries
    3. Arterioles
    4. Veins
    
    
#### Training Data Generation From H&E Images:
In order to avoid the generally large sizes of H&E images, and to provide the model with enough example images for training, a tiling approach was used to
break apart the larger image into smaller chunks to train the model. This was performed in QuPath where whole-slide annotations were present and overlaid over their corresponding regions in the original image. A script was used to tile the entire image using a specified tile size. Pairs of tiles consisting of the original image tile and its corresponding mask were saved and placed in the "imgs" and "masks" folders for training. It is a good idea to remember the tile size used during this process, as we will need it later to compute downsampling values during prediction.

## Methods:
1. Generate image pairs consisting of a multiplexed immunofluorescence (mIF) image and H&E image of a single whole-slide biopsy.

2. Create a QuPath project containing all image pairs, reserving an image pair for validation purposes.

3. Using the [Warpy Registration Package](https://imagej.net/plugins/bdv/warpy/warpy) in QuPath and ImageJ, register the mIF image to the H&E image to allow for later annotation transfer.

4. In QuPath, create a set of training annotations for the desired classes. Though dependent on what structures are being annotated, generally about 25 annotations are needed per class. These annotations may be split up across images as long as those images are still within the same project.

5. Train a QuPath Pixel Classifier using the set of manual annotations. Tweak the model parameters as needed to get a visually good result in "Live Prediction" mode. Further validation will be performed later on.

6. Using the holdout\validation image pair set asied in step 2, create a few rectangular annotations and fully annotate all structures inside them. These regions will be used as validation ground-truth annotations.

7. Apply the QuPath pixel classifier to the whole validation mIF image.

8. Validate the accuracy / F1 score of the predictions using the annotated validation regions. If the desired quality control metrics are poor, consider retraining.

9. Once satisfied with model performance, apply the QuPath pixel classifier to all other mIF images to generate whole-slide annotations that will be used for training the H&E segmentation classifier.

10. Using the image registration generated from step 3, transfer all of the annotations from the mIF images to their corresponding H&E image. Use the transform_objects.groovy script in the qupath_scripts folder to transfer the annotations. 

11. Using a tiling procedure, generate pairs of a raw image and a mask for each tile to be used as training images. Note that the size of the tiles may be image or tissue dependent. Aim to have a large enough tile size to be able to capture entire structures, but small enough to contain just a few structures. Use the export_labeled_tiles.groovy script in the qupath_scripts folder to export the pairs. Change the tile size and the requsted pixel size used for downscaling as needed.

12. Using a semantic segmentation classifier of your choice (in this case, DeepLabV3+ was used), feed the training images into the model. Tweak any parameters (tile size, input size, batch size, learning rate, ...etc.) as needed.

13. Using the trained model, predict on the whole holdout H&E image.

14. Calculate the accuracy \ F1 score (or any other QC metrics) using the same prediction rectangular regions.

15. If satisfied with the model's performance, use the model to predict on future H&E images of the tissue type trained on.
