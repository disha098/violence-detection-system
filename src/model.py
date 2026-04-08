from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def create_model():

    mobilenet = MobileNetV2(include_top=False , weights="imagenet")

    mobilenet.trainable = True

    for layer in mobilenet.layers[:-40]:
        layer.trainable = False

    model = Sequential()

    model.add(Input(shape=(16, 64, 64, 3)))

    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.25))

    # 🔥 IMPORTANT: ADD ALL THESE LAYERS (missing before)

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    return model


def load_trained_model():
    model = create_model()
    model.load_weights("models/model.weights.h5")
    return model