import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = float("inf")
import numpy as np
import argparse
from pathlib import Path

def getQCMetrics(truth_image_path,prediction_image_path, num_classes):
    
    truth_image = PIL.Image.open(truth_image_path)
    prediction_image = PIL.Image.open(prediction_image_path)
    
    if truth_image.size != prediction_image.size:
        print("Resizing Prediction Image of shape", prediction_image.size, "to Truth Image size of ", truth_image.size )
        prediction_image = prediction_image.resize(truth_image.size)
    
    truth_image = np.asarray(truth_image)
    prediction_image = np.asarray(prediction_image)
    
    image_size = prediction_image.size
    
    total_accuracy = (np.sum((truth_image == prediction_image)) / image_size) * 100
    
    val_dict = {curr_class : None for curr_class in range(1, num_classes)}
    for curr_class in range(1, num_classes):
        truth_class_image = truth_image == curr_class
        prediction_class_image = prediction_image == curr_class
        val_dict[curr_class] = (np.sum(truth_class_image == prediction_class_image) / image_size) * 100
    

    return total_accuracy, val_dict

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

    total_accuracy, val_dict = getQCMetrics(opts.truth_path,opts.prediction_path, opts.num_classes)
    
    print()
    print()
    print("------------------")
    print("Overall Accuracy:", total_accuracy)
    print()
    print("Class Accuracies:")
    print("------------------")
    if opts.class_names is None:
        for class_value in val_dict:
            print(class_value, ": ", val_dict[class_value], sep = "")
    else:
        for class_value, name in zip(val_dict.keys(), opts.class_names):
            print(name, ": ", val_dict[class_value], sep="")

if __name__=="__main__":
    main()