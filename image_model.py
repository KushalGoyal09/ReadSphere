import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import os
from glob import glob

# Load the pre-trained model and compiled it
model = load_model('model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the features extracted for all book covers
all_features = np.load('book_cover_features.npy')

# Path to the folder with downloaded images
download_folder = 'downloaded_images/'
image_paths = glob(os.path.join(download_folder, '*.jpg'))

def recommend_book(book_name):
    uploaded_image_path = os.path.join(download_folder, f"{book_name}.jpg")
    if not os.path.exists(uploaded_image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {uploaded_image_path}")
    user_image = cv2.imread(uploaded_image_path)
    if user_image is None:
        raise ValueError(f"Failed to load image from path: {uploaded_image_path}")
    user_image = cv2.resize(user_image, (224, 224))
    user_image = user_image / 255.0
    user_image = np.expand_dims(user_image, axis=0)
    user_features = model.predict(user_image)
    similarity_scores = cosine_similarity(user_features, all_features)
    similarity_scores = similarity_scores.flatten()
    top_n = 5
    similar_books_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    book_suggestions = []
    for idx in similar_books_indices:
        book_title = os.path.basename(image_paths[idx]).replace('.jpg', '')
        book_suggestions.append(book_title)