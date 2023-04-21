import argparse

import numpy as np
from keras.engine.saving import load_model

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from scripts.model import INPUT_SIZE


def build_test_generator(path, color_mode="rgb"):
    # create image data augmentation generator
    generator = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalization
    )

    generator = generator.flow_from_directory(
        directory=path,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        color_mode=color_mode,
        class_mode="categorical",
        shuffle=False,
        batch_size=32
    )

    return generator


# Testing a model on either one or two testing sets (named A and B)
def test(model, testing_path):
    test_generator = build_test_generator(testing_path)
    Y_pred_A = model.predict_generator(
        test_generator,
        steps=test_generator.samples // test_generator.batch_size+1
    )

    y_pred_A = np.argmax(Y_pred_A, axis=1)

    print('Confusion Matrix A')
    print(confusion_matrix(test_generator.classes, y_pred_A))

    print('Classification Report A')
    target_names = ['Left', 'Right', 'Straight']
    print(classification_report(test_generator.classes, y_pred_A, target_names=target_names))


#Load a model and test it on a testing set.
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--testing", type=str, required=True,
                    help="path to testing set of images")
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to model")
    args = vars(ap.parse_args())

    model = load_model(args["model"])
    test(model, args["testing"])