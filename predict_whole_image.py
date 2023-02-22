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
    predictions = model.predict(np.expand_dims((image_tensor), axis=0),verbose=0)
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def read_tile(image,input_size):
    image = tf.convert_to_tensor(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[input_size, input_size])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

def predict_tile(image_tensor, model):
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    return prediction_mask

def predict_and_stitch(scene, tile_size, model):
    
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
    whole_image = np.zeros(shape=(downY, downX , 3),dtype='uint8')
    prediction_mask = np.zeros(shape=(downY,downX),dtype='uint8')
    
    print("Tiling and Predicting")
    # Absolute coordinates
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

    
    print("Opening Image Viewer")
    view = napari.Viewer(show=False)
    view.add_image(whole_image,name="Image")
    view.add_labels(prediction_mask,
                    name="Predictions",
                    color=["red", "green", "blue", "purple", "cyan", "yellow", "orange"],
                    opacity=.65)
    view.show(block=True)
    
    print("Writing Image")
    savePath = Path("predictions")
    savePath.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(savePath / Path("model_image.png")), cv2.cvtColor(whole_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(savePath / Path("model_predictions.png")))


def get_opts():
    """"
    Handles the arguments needed to run the program. Note that the only non-optional arguments
    are the path to the training data and the path to save the model to.
    """
    parser = argparse.ArgumentParser(description = "Train A DeepLabV3+ Model Using Images & Masks")
    parser.add_argument('modelPath', type=str, help = "Path to the trained model from train.py")
    parser.add_argument('imagePath', type=str, help = "Path to folder containing image to tile and predict")
    parser.add_argument('tileSize', type=int, help = "Size of each tile to use as image to predict. Should be the same size as training tiles.")
    parser.add_argument('numClasses', type=int, help = "The number of classes to predict (should be the same as when training)")
    parser.add_argument('--outputPath', "-o", type=str, default = str(Path.cwd() / Path("/Predictions")), help = "Path to output prediction image")
    return parser.parse_args()

def main():
    opts = get_opts()

    try:
        model = tf.keras.models.load_model(opts.modelPath)
    except:
        print("Unable to load model using specified path", opts.modelPath)
        exit(1)
    
    try:
        image_type = opts.imagePath.split(".")[-1].upper()
        image = slideio.open_slide(opts.imagePath,image_type)
        scene = image.get_scene(0)
    except Exception as e:
        print("Unable to Load Image For Prediction", e)
        exit(1)

    predict_and_stitch(scene,opts.tileSize,model)

if __name__=="__main__":
    main()