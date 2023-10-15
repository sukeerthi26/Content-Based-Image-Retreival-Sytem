import os
from loguru import logger
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
import pickle 

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


class deepCBIR:
    def __init__(self):
        self.pickle_path = "./app/static/database1.pkl"          # To store feature vectors of images
        self.load_cbir_model()                                  
        self.vectorize_database("./app/dataset/training_set")   # Vectorizes the images in the training data

    def load_cbir_model(self):
        # Loads the pretrained model(InceptionResNetV2) for extracting features from image      
        self.cbir_model = InceptionResNetV2(weights="imagenet", include_top=True, input_shape=(299, 299, 3))    # Includes fully connected layer
        # Sets the model's last layer as the output
        self.last_layer = self.cbir_model.layers[-2].name
        self.cbir_model = Model(inputs=self.cbir_model.input, outputs=[self.cbir_model.get_layer(self.last_layer).output, self.cbir_model.output])

    def vectorize_database(self, database_dir):
        self.image_paths = []
        for foldername, subfolders, filenames in os.walk(database_dir):
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # Add the path of the image file to the list
                    self.image_paths.append(os.path.join(foldername, filename))
        
        # If there is saved pickle file, it loads features vectors from file
        try: 
            with open(self.pickle_path, "rb") as f:
                self.features, self.class_probabilities = pickle.load(f)

        # If not present, extracts features from images using InceptionResNetV2 model and saves it in pickle file
        except FileNotFoundError: 
            self.features = []
            self.class_probabilities = []
            for img_url in self.image_paths:
                print(img_url)
                try:
                    features, probabilities = self.extract_feature_and_probability(img_url, self.cbir_model)
                    self.features.append(features)
                    self.class_probabilities.append(probabilities)
                except:
                    pass
            with open(self.pickle_path, "wb") as f:
                pickle.dump((self.features, self.class_probabilities), f)

    # Resizes input image
    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)                   
        x = x / 255.0
        return x

    def extract_feature_and_probability(self, image_path, model):
        x = self.preprocess_image(image_path)
        features, probabilities = model.predict_on_batch(x)
        return features.flatten(), probabilities.flatten()

    def get_top_classes(self, probabilities, k=5):
        # Returns the top k classes with highest probabilities
        top_k_classes = probabilities.argsort()[-k:][::-1]
        return top_k_classes
    
    # Resizes input image
    def img_to_encoding(self, image_path, model):
        img1 = image.load_img(image_path, target_size=(299, 299, 3))
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        embedding = model.predict_on_batch(x)
        return embedding
    
    # Retrieve images based on using cosine similarity measure
    def retrieve_images(self, query_img_path, scope):
        # Extract features from query image
        query_features, query_probs = self.extract_feature_and_probability(query_img_path, self.cbir_model)

        # Assign top 5 classes to each image in the database based on probability output
        top_classes = []
        for probs in self.class_probabilities:
            top_classes.append(np.argsort(probs)[::-1][:5])
    
        # Assign top 5 classes to query image based on probability output
        query_classes = np.argsort(query_probs)[::-1][:5]
    
        # Find images in the database that belong to one of the query's top classes
        similar_images = []
        for i, top_classes in enumerate(top_classes):
            if any(c in top_classes for c in query_classes):
                # Calculate similarity between query and database images
                similarity = cosine_similarity(query_features.reshape(1, -1), self.features[i].reshape(1, -1)).ravel()
                similar_images.append((self.image_paths[i], similarity))
    
        # Sort the similar images by descending similarity
        similar_images.sort(key=lambda x: x[1], reverse=True)

        image_paths=[]
        for image_path, image_similarity in similar_images[:scope]:
            image_paths.append(image_path)

        return image_paths
    
    # To display images in grid format
    def create_plot(self, image_paths):
        if len(image_paths) == 1:
            img = Image.open(image_paths[0])
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_title('Query Image')
            ax.axis("off")
            fig.savefig("./app/tmp/query.jpg")
        else:
            rows = (len(image_paths) // 5)
            if len(image_paths) % 5 != 0:
                rows += 1
            if rows == 1:
                cols = len(image_paths)
            else:
                cols = 5
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5))
            fig.suptitle('Retrieved Images', fontsize=20)
            for i in range(len(image_paths)):
                x = i % 5
                y = i // 5
                img = Image.open(image_paths[i])
                img = img.resize((299, 299))
                if rows == 1:
                    axes[x].imshow(img)
                    axes[x].axis("off")
                else:
                    axes[y, x].imshow(img)
                    axes[y, x].axis("off")

            fig.savefig("./app/tmp/retrieved.jpg")