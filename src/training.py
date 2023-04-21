import argparse

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

from scripts.model import INPUT_SIZE
from scripts.model import build_model


# Data generator for training that produces data augmented versions of training images.
def build_train_generator(path, color_mode="rgb"):
    generator = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalization
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    generator = generator.flow_from_directory(
        directory=path,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        color_mode=color_mode,
        class_mode="categorical",
        shuffle=True,
        batch_size=32
    )

    return generator


# Load and train a model.
def train_from_path(model_path, training_path, validation_path, name="model"):
    model = load_model(model_path)
    train(model_path, training_path, validation_path, name="model")


# Train a loaded model. Includes early stopping mechanics and checkpoints.
def train(model, training_path, validation_path, name="model"):
    train_generator = build_train_generator(training_path)
    validation_generator = build_train_generator(validation_path)

    early_stop = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=2)
    checkpoint = ModelCheckpoint("../" + name + ".h5", monitor='val_accuracy', mode='max', verbose=1,
                                 save_best_only=True)
    history_model = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[checkpoint, early_stop]
    )

    #write history
    pd.DataFrame.from_dict(history_model.history).to_csv('../history_' + name + '.csv', index=False)


#Load and train a model
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", type=str, required=True,
                    help="path to training set of images")
    ap.add_argument("-v", "--validation", type=str, required=True,
                    help="path to validation set of images")
    ap.add_argument("-m", "--model", type=str, default="None",
                    help="path to trained model")
    ap.add_argument("-n", "--name", type=str, default="model",
                    help="model name")
    args = vars(ap.parse_args())

    model = None
    if(args["model"] is None):
        model = build_model()
    else:
        model = load_model(args["model"])
    train(model, args["training"], args["validation"], args["name"])
