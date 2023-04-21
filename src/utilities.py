import time
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import pandas as pd


# Add prediction GUI elements to an image
def add_gui(img, prediction):
    img = show_direction_arrow(img, prediction)
    img = show_absolute_direction_arrow(img, prediction)

    return img


def show_direction_arrow(img, direction_confidences):
    radian = math.radians(average_angle(direction_confidences))
    h, w, _ = img.shape

    arrow_length = h//4

    start_point = (w//2, h)
    end_point = (round(w//2 + (arrow_length*-math.cos(radian))), round(h-(arrow_length*math.sin(radian))))
    color = (0, 0, 255) #red
    thickness = math.ceil(arrow_length//30)

    img = cv2.arrowedLine(img, start_point, end_point, color, thickness)
    return img


def average_angle(direction_confidences):
    confidences = direction_confidences[0]
    right = confidences[1]
    left = confidences[2]
    confidence_sum = sum(confidences)
    average = (180*right + 90*left) / confidence_sum
    return average


def show_absolute_direction_arrow(img, direction_confidences):
    angle = 0
    if direction_confidences[0][0] == np.amax(direction_confidences):
        angle = 45
    elif direction_confidences[0][2] == np.amax(direction_confidences):
        angle = 90
    else:
        angle = 135

    radian = math.radians(angle)
    h, w, _ = img.shape

    arrow_length = h//4
    start_point = (w//2, h)
    end_point = (round(w//2 + (arrow_length*-math.cos(radian))), round(h-(arrow_length*math.sin(radian))))

    color = (0, 255, 0) #red
    thickness = math.ceil(arrow_length//30)

    img = cv2.arrowedLine(img, start_point, end_point, color, thickness)
    return img



# Measures time (in s) for a model to make a prediction on an image.
def time_to_compute(model, img):
    timer = time.process_time()
    model.predict(img)
    timer = time.process_time() - timer
    print("Time: ", timer, " s" )



# Displays graph for a history .cvs file.
def history_visualisation(history_path):
    history = pd.read_csv(open(history_path, 'r')).to_dict()
    plt.plot(list(history['accuracy'].values()))
    plt.plot(list(history['val_accuracy'].values()))
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()