import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input


class AlexNet:
    def __init__(self, input_shape=(224, 224, 1), num_classes=1000):
        input1 = Input(shape=input_shape)
        conv1 = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(input1)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

        conv2 = Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
            
        conv3 = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
        conv4 = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
        pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv5)
            
        flat1 = Flatten()(pool3)
        dense1 = Dense(4096, activation='relu')(flat1)
        drop1 = Dropout(0.5)(dense1)
        output1 = Dense(4096, activation='relu')(drop1)  # Feature extraction layer
        drop2 = Dropout(0.5)(output1)
        output2 = Dense(num_classes, activation='softmax')(drop2)  # Classification layer

        # Create the full model
        self.model = Model(inputs=input1, outputs=[output1, output2])
        self.model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

        # Create the feature extraction model
        self.feature_extractor = Model(inputs=input1, outputs=output1)
        self.feature_extractor.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

        # Create the classification model
        self.classifier = Model(inputs=input1, outputs=output2)
        self.classifier.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])


class VGG16:
    def __init__(self, input_shape=(224, 224, 1), num_classes=1000):
        input1 = Input(shape=input_shape)
        conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(input1)
        conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        
        conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

        conv5 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
        conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
        conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv7)

        conv8 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
        conv9 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv8)
        conv10 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv9)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv10)

        conv11 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
        conv12 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv11)
        conv13 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv12)
        pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv13)

        flat1 = Flatten()(pool5)
        dense1 = Dense(4096, activation='relu')(flat1)
        drop1 = Dropout(0.5)(dense1)
        output1 = Dense(4096, activation='relu')(drop1)  # Feature extraction layer
        drop2 = Dropout(0.5)(output1)
        output2 = Dense(num_classes, activation='softmax')(drop2)  # Classification layer
        
        # Create the full model
        self.model = Model(inputs=input1, outputs=[output1, output2])
        self.model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

        # Create the feature extraction model
        self.feature_extractor = Model(inputs=input1, outputs=output1)
        self.feature_extractor.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

        # Create the classification model
        self.classifier = Model(inputs=input1, outputs=output2)
        self.classifier.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

