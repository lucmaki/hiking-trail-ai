from keras.applications.mobilenet_v2 import MobileNetV2


INPUT_SIZE = 128    #Size for model input layer + for resizing images


#Returns a new compiled MobileNetV2 model
def build_model():
    model = MobileNetV2(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), weights=None, classes=3)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
