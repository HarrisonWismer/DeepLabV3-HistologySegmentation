import os
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

import tensorflow as tf
from tensorflow import keras
from keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DLV3Model():
    
    def __init__(self, image_size, num_classes, val_split, batch_size, learning_rate, num_epochs):
        self.image_size = image_size
        self.num_classes = num_classes
        self.val_split = val_split
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.model = self.DeeplabV3Plus(image_size,num_classes)

    def convolution_block(self,block_input, num_filters=256,kernel_size=3,dilation_rate=1,padding="same",use_bias=False,):
        x = layers.Conv2D(num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),)(block_input)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x)


    def DilatedSpatialPyramidPooling(self,dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",)(x)

        out_1 = self.convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, kernel_size=1)
        return output
    
    def DeeplabV3Plus(self,image_size, num_classes):
        model_input = keras.Input(shape=(image_size, image_size, 3))
        resnet50 = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)
        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
        return keras.Model(inputs=model_input, outputs=model_output)
    
    def train_model(self, input_path):
        
        train_loader = DataLoader(input_path, self.image_size, self.batch_size, self.val_split)
        train_dataset, val_dataset = train_loader.load_training_data()
        
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=["accuracy"],)

        history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.num_epochs)
        
        training_loss = history.history['loss'][-1]
        training_accuracy = history.history['accuracy'][-1]
        validation_loss = history.history['val_loss'][-1]
        validation_accuracy = history.history['val_accuracy'][-1]
        print("Training Loss:", training_loss)
        print("Training Accuracy:", training_accuracy)
        print("Validation Loss:", validation_loss)
        print("Validation Accuracy", validation_accuracy)
    
    def get_model(self):
        return self.model
    
    def save_model(self, path):
        try:
            self.model.save(path)
        except:
            print("Invalid Path For Saving Model")
            
class DataLoader():
    
    def __init__(self,input_path, image_size, batch_size, val_split):
        self.image_size = image_size
        self.input_path = input_path
        self.batch_size = batch_size
        self.val_split = val_split

    def read_image(self,image_path, mask=False):
        image = tf.io.read_file(image_path)
        if mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
            image = tf.keras.applications.resnet50.preprocess_input(image)
        return image


    def load_data(self,image_list, mask_list):
        image = self.read_image(image_list)
        mask = self.read_image(mask_list, mask=True)
        return image, mask


    def data_generator(self,image_list, mask_list):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset
    
    def load_training_data(self):
        image_paths = sorted(glob(os.path.join(self.input_path, "imgs\\*")))
        mask_paths = sorted(glob(os.path.join(self.input_path, "masks\\*")))

        X_train, X_test, y_train, y_test = train_test_split(image_paths, mask_paths, 
                                                            test_size=self.val_split, 
                                                            random_state=42,
                                                            shuffle=True)
        
        train_dataset = self.data_generator(X_train, y_train)
        val_dataset = self.data_generator(X_test, y_test)
        print("Train Dataset:", train_dataset)
        print("Val Dataset:", val_dataset)
        
        return train_dataset,val_dataset


def get_opts():
    parser = argparse.ArgumentParser(description = "Train A DeepLabV3+ Model Using Images & Masks")
    parser.add_argument('--imageSize', '-i', type=int, default = 256, help = "n x n Image Size To Downscale Images To. Default = 256")
    parser.add_argument('--numClasses', '-n', type=int, default = 2, help = "Number of Classes. Default = 2")
    parser.add_argument('--valSplit', '-v', type=float, default = .2, help = "Proportion of Training Data To Be Used For Validation. Default = .2")
    parser.add_argument('--batchSize', '-b', type=int, default=8, help = "Batch Size. Default = 8")
    parser.add_argument('--learningRate', '-l', type=float, default=.0001, help = "Learning Rate. Default = .0001")
    parser.add_argument('--numEpochs', '-e', type=int, default = 5, help = "Numer of Epochs. Default = 5")
    parser.add_argument('trainingPath', type=str, help = "Folder containing Training Images & Masks.")
    parser.add_argument('savePath',type=str, default = str(Path.cwd()), help = "Path To Save Trained Model To.")

    return parser.parse_args()

def main():
    opts = get_opts()

    print(opts.trainingPath)

    if not os.path.exists(opts.trainingPath) or not os.path.isdir(opts.trainingPath) or opts.trainingPath is None:
        print("Invalid Traning Path")
        exit(1)
    
    if not os.path.exists(opts.savePath):
        print("Creating Directory:", opts.savePath, "for saving")
        os.makedirs(opts.savePath)

    print("Creating Model")
    myDLV3 = DLV3Model(image_size = opts.imageSize, 
                   num_classes= opts.numClasses, 
                   val_split = opts.valSplit, 
                   batch_size = opts.batchSize, 
                   learning_rate = opts.learningRate,
                   num_epochs = opts.numEpochs)
    print("Loading Training Data")
    myDLV3.train_model(opts.trainingPath)
    myDLV3.save_model(opts.savePath)

if __name__=="__main__":
    main()