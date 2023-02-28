import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = float("inf")
import numpy as np
import argparse
from pathlib import Path

def getQCMetrics(truth_image_path,prediction_image_path, num_classes):
    """
    Given the paths to a truth label-image, a prediction label image, and the number of classes, output quality control metrics.

    Input:
        truth_image_path -> Path to an image with ground truth labels
        prediction_image_path -> Path to an image with prediction labels
        num_classes -> The total number of classes present (background (0) counts as a class))

    Output:
        An F1-Score for each class.
    """
    # Read in image paths into PILLOW images
    truth_image = PIL.Image.open(truth_image_path)
    prediction_image = PIL.Image.open(prediction_image_path)

    # If the masks are different sizes (one is downsampled etc.) resize the prediction image to the same size as the truth image
    if truth_image.size != prediction_image.size:
        print("Resizing Prediction Image of shape", prediction_image.size, "to Truth Image size of ", truth_image.size )
        prediction_image = prediction_image.resize(truth_image.size)
    
    # Cast both to np arrays for easily manipulation
    truth_image = np.asarray(truth_image)
    prediction_image = np.asarray(prediction_image)

    val_dict = {curr_class : None for curr_class in range(1, num_classes)}

    for curr_class in range(1,num_classes):
        true_positives = (truth_image == curr_class) & (prediction_image == curr_class) & (truth_image == prediction_image)
        false_positives = (prediction_image == curr_class) & (truth_image != curr_class)
        false_negatives = (truth_image == curr_class) & (prediction_image != curr_class)

        f1 = (2 * np.sum(true_positives)) / ((2*np.sum(true_positives)) + np.sum(false_positives) + np.sum(false_negatives))
        val_dict[curr_class] = f1

    return val_dict

def get_opts():
    """"
    Handles the arguments needed to run the program. Note that the only non-optional argument is whether
    to show the viewer to interactively view predictions.
    """
    parser = argparse.ArgumentParser(description = "Get Accuracy Statistics for a prediction label-image.")
    parser.add_argument('truth_path', type=str, help = "Path to label-image with ground truth values.")
    parser.add_argument('prediction_path', type=str, help = "Path to the labe-image with prediction values.")
    parser.add_argument('num_classes', type=int, help = "The number of classes in the ground truth image")
    parser.add_argument('--class-names', '-c', type=str, nargs ="+", help = "Input class names.")
    return parser.parse_args()

def main():
    opts = get_opts()

    if not Path.exists(Path(opts.truth_path)):
        print("Cannot Read Ground Truth Image")
        exit(1)
    
    if not Path.exists(Path(opts.prediction_path)):
        print("Cannot Read Prediction Image")
        exit(1)

    print()
    val_dict = getQCMetrics(opts.truth_path,opts.prediction_path, opts.num_classes)
    
    print()
    print("------------------")
    print("Class F1 Scores:")
    print("------------------")
    if opts.class_names is None:
        for class_value in val_dict:
            print(class_value, ": ", val_dict[class_value], sep = "")
    else:
        for class_value, name in zip(val_dict.keys(), opts.class_names):
            print(name, ": ", val_dict[class_value], sep="")

if __name__=="__main__":
    main()