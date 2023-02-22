import cv2
import numpy as np
from pathlib import Path
import slideio
import argparse
import napari

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def infer(model, image_tensor):
    """
    Returns an array of predictions for a given image.
    See: https://keras.io/examples/vision/deeplabv3_plus/
    """
    predictions = model.predict(np.expand_dims((image_tensor), axis=0),verbose=0)
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def read_tile(image,input_size):
    """
    Given an image and the input size of the model, read and pre-process the image.
    See: https://keras.io/examples/vision/deeplabv3_plus/
    """
    image = tf.convert_to_tensor(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[input_size, input_size])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

def predict_tile(image_tensor, model):
    """
    Returns the prediction mask for a given tile.
    """
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    return prediction_mask

def predict_and_stitch(scene, tile_size, model, viewer = True):
    """

    Given an entire image, tiles the image and uses to model to predict on each tile. The whole image is then reconstructed
    tile by tile to create a whole image prediction overlay. Napari will then open and show the original image with the prediction
    overlay.

    Input:
        - scene: An entire image
        - tile_size: The size of each tile to be taken from the image. Should be the same as the tile size used for training.
        - model: The model previously trained in train.py

    Output:
        - scaled_image: The image scaled down by tile_size / input_size
        - prediction_overlay: The prediction overlay, with the same size as the scaled_image
    """
    
    # Size of Original Image
    x,y = scene.rect[2:]
    
    # Calculate downscaling factor for prediction image.
    input_size = model.layers[0].input_shape[0][1]
    downscale = tile_size / input_size
    
    # Crop image to fit the exact tile size without extending past the image.
    origXStart = x % tile_size
    origYStart = y % tile_size
    downX = int((x - origXStart) // downscale)
    downY = int((y - origYStart) // downscale)

    print("Allocating Image Space")
    whole_image = np.zeros(shape=(downY, downX , 3),dtype='uint8') # RGB Image
    prediction_mask = np.zeros(shape=(downY,downX),dtype='uint8') # Label Image
    
    print("Tiling and Predicting")
    for rectY in range(origYStart, y, tile_size):
        for rectX in range(origXStart, x, tile_size):

            downRectX = int(rectX / downscale)
            downRectY = int(rectY / downscale)

            image = scene.read_block((rectX,rectY,tile_size,tile_size))
            scaled_image = cv2.resize(image,dsize=(input_size,input_size))
            image_tile = read_tile(image,input_size)
            prediction_overlay = predict_tile(image_tile, model)
            
            try:
                whole_image[downRectY:downRectY+input_size, downRectX:downRectX+input_size, :] = scaled_image
                prediction_mask[downRectY:downRectY+input_size, downRectX:downRectX+input_size] = prediction_overlay
            except:
                continue

    if viewer == True:
        colors = {1: "red", 2:"green", 3: "blue", 4:"purple"}
        # Open viewer to see predictions in napari
        print("Opening Image Viewer")
        view = napari.Viewer(show=False)
        view.add_image(whole_image,name="Image")
        view.add_labels(prediction_mask,
                        name="Predictions",
                        opacity=.65,
                        color=colors)
        view.show(block=True)

    return whole_image, prediction_mask


def get_opts():
    """"
    Handles the arguments needed to run the program. Note that the only non-optional argument is whether
    to show the viewer to interactively view predictions.
    """
    parser = argparse.ArgumentParser(description = "Train A DeepLabV3+ Model Using Images & Masks")
    parser.add_argument('model_path', type=str, help = "Path to the trained model from train.py")
    parser.add_argument('image_path', type=str, help = "Path to folder containing image to tile and predict")
    parser.add_argument('tile_size', type=int, help = "Size of each tile to use as image to predict. Should be the same size as training tiles.")
    parser.add_argument('num_classes', type=int, help = "The number of classes to predict (should be the same as when training)")
    parser.add_argument('--show-viewer', '-v', type=bool, default = True, help = "Show the predictions in an interactive napari viewer session.")
    return parser.parse_args()

def main():
    opts = get_opts()

    try:
        model = tf.keras.models.load_model(opts.model_path)
    except:
        print("Unable to load model using specified path", opts.model_path)
        exit(1)
    
    try:
        image_type = opts.image_path.split(".")[-1].upper()
        image = slideio.open_slide(opts.image_path,image_type)
        scene = image.get_scene(0)
    except Exception as e:
        print("Unable to Load Image For Prediction", e)
        exit(1)

    scaled_image, prediction_overlay = predict_and_stitch(scene, opts.tile_size, model, opts.show_viewer)

    print("Writing Images")
    image_name = opts.image_path.split(".")[0]
    savePath = Path("predictions"); savePath.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(savePath / Path(image_name + "_image.png")), cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(savePath / Path(image_name + "_predictions.png")), prediction_overlay)

if __name__=="__main__":
    main()