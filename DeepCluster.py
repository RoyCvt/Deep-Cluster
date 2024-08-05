import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from faiss import PCAMatrix, Kmeans
from ConvNets import AlexNet, VGG16
import random
import cv2
import os

class DeepCluster:
    def __init__(self, model_name='alexnet', input_shape=(224, 224, 1), num_classes=1000, dataset_path='Dataset/'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.feature_extractor = None
        self.classifier = None

        # Define the encoder model (AlexNet by default)
        if model_name == 'alexnet':
            alexnet = AlexNet(input_shape=self.input_shape, num_classes=self.num_classes)
            self.feature_extractor = alexnet.feature_extractor
            self.classifier = alexnet.classifier
        elif model_name == 'vgg16':
            vgg16 = VGG16(input_shape=self.input_shape, num_classes=self.num_classes)
            self.feature_extractor = vgg16.feature_extractor
            self.classifier = vgg16.classifier
        else:
            raise ValueError(f"model_name must be 'alexnet' or 'vgg16'. Got {model_name}")

        # Optimizer for training the final classifier (optional for this example)
        self.classifier_optimizer = Adam()


    def load_data(self, image_limit):
        """
        Returns a list of images from a dataset.

        Args:
            limit: Maximum amount of images to take from each class of the dataset.

        Returns:
            A list of at most n*limit images where n is the amount of classes in the dataset.
        """
        data = []

        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            for img_num, img_name in enumerate(os.listdir(class_path)):
                try:
                    # Take only "limit" amount of images from each subfolder
                    if img_num >= image_limit:
                        break

                    # Path to the current image
                    img_path = os.path.join(class_path, img_name)
                    # Content of the image (np array)
                    img_content = cv2.imread(img_path)
                    # Reshape images 
                    img_content = cv2.resize(img_content, self.input_shape[:2])
                    # Add to list of images
                    data.append(img_content)
                except Exception as e:
                    print(e)

        return data


    def shuffle_data(self, data, times=5):
        for i in range(times):
            random.shuffle(data)
        return data 


    def preprocess_data(self, data):
        """
        Applies random augmentations to data.

        Args:
            data: A an list of images (list of np arrays).

        Returns:
            A list of modified images.
        """

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Apply augmentations to each image
        augmented_data = []
        for img in data:
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            # Create augmentation iterator
            aug_iter = datagen.flow(img, batch_size=1)
            # Augment image
            augmented_img = next(aug_iter)
            # Remove batch dimension 
            augmented_img = augmented_img[0]
            # Add image to list of augmented data
            augmented_data.append(augmented_img)

        # Preprocessing: Grayscale and Sobel
        preprocessed_data = []
        for img in augmented_data:
            # Convert to grayscale
            img_gray = tf.image.rgb_to_grayscale(img)
            # Add batch dimension
            img_gray = tf.expand_dims(img_gray, axis=0)
            # Apply the Sobel operator to the image
            img_sobel = tf.image.sobel_edges(img_gray)
            # Remove batch dimension
            img_sobel = img_sobel[0]
            # Combine gradient magnitudes for a single channel
            img_sobel = tf.math.sqrt(tf.square(img_sobel[..., 0]) + tf.square(img_sobel[..., 1]))

            # Normalize to 0-1 range
            img_sobel = np.asarray(img_sobel, dtype=np.float32)
            img_sobel_normalized = (img_sobel - np.min(img_sobel)) / (np.max(img_sobel) - np.min(img_sobel))

            # Add image to list of pre-processed data
            preprocessed_data.append(img_sobel_normalized)

        return preprocessed_data


    def extract_features(self, data):
        data_array = np.asarray(data)
        features = self.feature_extractor.predict(data_array)
        return features


    def normalize_features(self, features, output_dim=256):
        # Apply PCA for dimensionality reduction 
        input_dim = features.shape[1]
        pca = PCAMatrix(input_dim, output_dim, eigen_power=-0.5)
        pca.train(features)
        features = pca.apply_py(features)
        # L2 normalization
        normalized_features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
        return normalized_features
    

    def cluster_features(self, features, num_clusters):
        # Perform K-means clustering on the reduced features
        feature_dim = features.shape[1]
        kmeans = Kmeans(feature_dim, num_clusters)
        kmeans.train(features)
        _, I = kmeans.index.search(features, 1)
        cluster_labels = [i[0] for i in I]
        return cluster_labels


    def train(self, epochs, batch_size):
        data = self.load_data(500)

        for i in range(epochs):
            # Data shuffling
            shuffled_data = self.shuffle_data(data)
            # Data pre-processing
            preprocessed_data = self.preprocess_data(shuffled_data)
            # Feature extraction
            features = self.extract_features(preprocessed_data)
            # Dimensionality reduction and normalization of features 
            normalized_features = self.normalize_features(features)
            # Clustering
            cluster_labels = self.cluster_features(features=normalized_features, num_clusters=self.num_classes)

            X_train = np.asarray(preprocessed_data)
            y_train = np.asarray(cluster_labels)

            self.classifier.fit(X_train, y_train, epochs=1, batch_size=batch_size)
            

        

def main():
    train_path='Dataset/train/'
    deep_cluster = DeepCluster(model_name='alexnet', input_shape=(224, 224, 1), num_classes=2, dataset_path=train_path)
    deep_cluster.train(epochs=100, batch_size=256)
    

if __name__ == "__main__":
    main()
