import os
import warnings

# -------------------------------
# SUPPRESS WARNINGS AND TF LOGS
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow INFO & WARNING logs
warnings.filterwarnings("ignore")          # Suppress Python warnings

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(Conv2D(96, (11,11), strides=4, activation='relu', input_shape=input_shape))
        self.add(MaxPooling2D((3,3), strides=2))

        self.add(Conv2D(256, (5,5), padding='same', activation='relu'))
        self.add(MaxPooling2D((3,3), strides=2))

        self.add(Conv2D(384, (3,3), padding='same', activation='relu'))
        self.add(Conv2D(384, (3,3), padding='same', activation='relu'))
        self.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.add(MaxPooling2D((3,3), strides=2))

        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax'))


if __name__ == "__main__":
    model = AlexNet((224, 224, 3), 1000)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
