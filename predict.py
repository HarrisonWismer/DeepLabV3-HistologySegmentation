import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import slideio
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def predict_and_stitch(scene, tile_size, colormap, model):
    
    # Get the Model's input size and the x,y size of the entire image
    input_size = model.layers[0].input_shape[0][1]
    x,y = scene.rect[2:]
    
    # Calculate downscaling factor for prediction image.
    downscale = tile_size / input_size
    downX =  int(x // downscale)
    downY = int(y // downscale)
    
    print("Allocating Image Space")
    wholeImage = np.zeros(shape=(downY, downX , 3),dtype='uint8')
    
    origXStart = x % tile_size
    origYStart = y % tile_size
    
    print("Tiling and Predicting")
    # Absolute coordinates
    for rectY in range(origYStart, y, tile_size):
            
        for rectX in range(origXStart, x, tile_size):

            downRectX = int(rectX / downscale)
            downRectY = int(rectY / downscale)

            image = scene.read_block((rectX,rectY,tile_size,tile_size))
            image = read_tile(image)
            prediction_overlay = predict_tile(image,colormap,model)

            
            wholeImage[downRectY:downRectY+input_size, downRectX:downRectX+input_size, :] = prediction_overlay
    
    
    print("Writing Image to", str(image_path) + "_predictions.png")
    savePath = Path("Predictions")
    savePath.mkdir(parents=True, exist_ok=True)
    fname = image_path.name.split(".svs")[0]
    cv2.imwrite(str(savePath / Path(fname + "_predictions.png")), cv2.cvtColor(wholeImage, cv2.COLOR_RGB2BGR))


def get_opts():
    """"
    Handles the arguments needed to run the program. Note that the only non-optional arguments
    are the path to the training data and the path to save the model to.
    """
    parser = argparse.ArgumentParser(description = "Train A DeepLabV3+ Model Using Images & Masks")
    parser.add_argument('modelPath', type=str, help = "Path to the trained model from train.py")
    parser.add_argument('imagePath', type=str, help = "Path to folder containing image to tile and predict")
    parser.add_argument('tileSize', type=int, help = "Size of each tile to use as image to predict. Should be the same size as training tiles.")
    parser.add_argument('--outputPath', "-o", type=str, default = str(Path.cwd()), help = "Path to output prediction image")
    return parser.parse_args()

def main():
    opts = get_opts()

    print(opts.trainingPath)

    try:
        model = tf.keras.models.load_model(opts.modelPath)
    except:
        print("Unable to load model using specified path", opts.modelPath)
        exit(1)
    
    try:
        image_type = opts.imagePath.split(".")[-1]
        image = slideio.open_slide(opts.imagePath,image_type)
        scene = image.get_scene(0)
    except Exception as e:
        print("Unable to Load Image For Prediction", e)



    

if __name__=="__main__":
    main()