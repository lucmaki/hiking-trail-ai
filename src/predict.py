import argparse
from glob import glob

import numpy as np
import cv2
from keras.engine.saving import load_model

from scripts.model import INPUT_SIZE
from scripts.utilities import average_angle, add_gui

#Iterate through a folder to predict each image
def predict_folder(model, folder_path, wait_time=0):
    print("Indices: ", "{'left': 0, 'right': 1, 'straight': 2}")

    for img_path in glob(folder_path):
        predict(model, img_path, wait_time)


#Predict an image from a model, and display results, then wait a certain time (default is 0: waits for key press)
def predict(model, img_path, wait_time=0):
    color_mode = cv2.IMREAD_COLOR

    og_img = cv2.imread(img_path, color_mode)
    img = cv2.imread(img_path, color_mode)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = np.expand_dims(img, axis=0)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    prediction = model.predict(img)

    print(img_path)
    print(prediction)
    print("Angle: ", average_angle(prediction))

    og_img = add_gui(og_img, prediction)

    cv2.imshow("Prediction",og_img)
    cv2.waitKey(wait_time)

#Load a model and predict a list of images.
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, required=True,
                    help="path to set of images to predict")
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to model")
    args = vars(ap.parse_args())

    path = args["images"] + "/*.jpg"

    model = load_model(args["model"])
    predict_folder(model, path)