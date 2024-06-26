import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import napari
import argparse

def get_opts():
    """"
    Handles the arguments needed to run the program. Note that the only non-optional argument is whether
    to show the viewer to interactively view predictions.
    """
    parser = argparse.ArgumentParser(description = "Train A DeepLabV3+ Model Using Images & Masks")
    parser.add_argument('image_path', type=str, help = "Path to the downscaled image generated from predict_whole_image.py")
    parser.add_argument('overlay_path', type=str, help = "Path to the prediction mask generated from predict_whole_image.py")
    return parser.parse_args()

def main():
    """
    This script relies on the ouput from the predict_whole_image.py script. All this script does is use the image
    and mask generated in predict_whole_image.py and overlays them in napari. This allows interactive viewing.
    
    """
    opts = get_opts()

    view = napari.Viewer(show=False)
    colors = {1:"red", 2:"green", 3:"blue", 4:"purple"}

    try:
        img = cv2.imread(opts.image_path, cv2.COLOR_BGR2RGB)
        view.add_image(img, name = "Image")
    except Exception as e:
        print("Unable to open image")
        print(e)
        exit(1)
    
    try:
        mask = cv2.imread(opts.overlay_path, cv2.IMREAD_GRAYSCALE)
        view.add_labels(mask,
                        name = "Predictions",
                        color=colors,
                        opacity=.65)
    except Exception as e:
        print("Unable to open mask")
        print(e)
        exit(1)
    
    view.show(block=True)


if __name__=="__main__":
    main()