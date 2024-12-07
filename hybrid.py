import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os
from glob import glob

books = pd.read_csv("books.csv", sep=";", on_bad_lines='skip', encoding='latin-1', low_memory=False)
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher'}, inplace=True)

# Load and preprocess users data
users = pd.read_csv("users.csv", sep=";", on_bad_lines='skip', encoding='latin-1')
users.rename(columns={"User-ID": 'user_id', "Location": "location", "Age": "age"}, inplace=True)

# Load and preprocess ratings data
ratings = pd.read_csv('ratings.csv', sep=";", on_bad_lines='skip', encoding='latin-1')
ratings.rename(columns={"User-ID": "user_id", "Book-Rating": "rating"}, inplace=True)

# Filter users with more than 200 ratings
x = ratings['user_id'].value_counts() > 200
y = x[x].index
ratings = ratings[ratings['user_id'].isin(y)]

# Merge ratings with book details
ratings_with_books = ratings.merge(books, on="ISBN")

# Aggregate ratings count per book title
number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating': 'number_of_ratings'}, inplace=True)

# Filter for books with more than 50 ratings and remove duplicates
final_rating = ratings_with_books.merge(number_rating, on='title')
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

class HybridBookRecommender:
    def __init__(self, image_weight=0.3, rating_weight=0.7):
        """
        Initialize the hybrid recommender with weights for each model.
        
        Args:
            image_weight (float): Weight for image-based recommendations (0-1)
            rating_weight (float): Weight for rating-based recommendations (0-1)
        """
        self.image_weight = image_weight
        self.rating_weight = rating_weight
        self.setup_image_model()
        self.setup_rating_model()
    
    def setup_image_model(self):
        """Initialize the CNN-based image recommendation model"""
        self.cnn_model = load_model('model.h5')
        self.cnn_model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
        self.all_features = np.load('book_cover_features.npy')
        self.download_folder = 'downloaded_images/'
        self.image_paths = glob(os.path.join(self.download_folder, '*.jpg'))
    
    def setup_rating_model(self):
        """Initialize the KNN-based rating recommendation model"""
        # Load and preprocess data
        self.books = pd.read_csv("books.csv", sep=";", 
                               on_bad_lines='skip', 
                               encoding='latin-1', 
                               low_memory=False)
        self.books = self.books[['ISBN', 'Book-Title', 'Book-Author', 
                                'Year-Of-Publication', 'Publisher']]
        self.books.rename(columns={
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Year-Of-Publication': 'year',
            'Publisher': 'publisher'
        }, inplace=True)
        
        ratings = pd.read_csv('ratings.csv', sep=";", 
                            on_bad_lines='skip', 
                            encoding='latin-1')
        ratings.rename(columns={
            "User-ID": "user_id",
            "Book-Rating": "rating"
        }, inplace=True)
        
        # Filter active users
        active_users = ratings['user_id'].value_counts() > 200
        active_user_ids = active_users[active_users].index
        ratings = ratings[ratings['user_id'].isin(active_user_ids)]
        
        # Create final rating matrix
        ratings_with_books = ratings.merge(self.books, on="ISBN")
        number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
        number_rating.rename(columns={'rating': 'number_of_ratings'}, inplace=True)
        
        final_rating = ratings_with_books.merge(number_rating, on='title')
        final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
        final_rating.drop_duplicates(['user_id', 'title'], inplace=True)
        
        # Create pivot table and fit KNN model
        self.book_pivot = final_rating.pivot_table(
            columns='user_id',
            index='title',
            values='rating'
        ).fillna(0)
        
        book_sparse = csr_matrix(self.book_pivot)
        self.knn_model = NearestNeighbors(algorithm='brute')
        self.knn_model.fit(book_sparse)
    
    def get_image_recommendations(self, book_name):
        """Get recommendations based on book cover similarity"""
        try:
            image_path = os.path.join(self.download_folder, f"{book_name}.jpg")
            if not os.path.exists(image_path):
                return []
            
            user_image = cv2.imread(image_path)
            if user_image is None:
                return []
            
            user_image = cv2.resize(user_image, (224, 224))
            user_image = user_image / 255.0
            user_image = np.expand_dims(user_image, axis=0)
            
            user_features = self.cnn_model.predict(user_image)
            similarity_scores = cosine_similarity(user_features, self.all_features)
            similarity_scores = similarity_scores.flatten()
            
            similar_books_indices = similarity_scores.argsort()[-6:-1][::-1]
            return [os.path.basename(self.image_paths[idx]).replace('.jpg', '')
                   for idx in similar_books_indices]
        except Exception as e:
            print(f"Error in image recommendations: {str(e)}")
            return []
    
    def get_rating_recommendations(self, book_name):
        """Get recommendations based on user ratings"""
        try:
            book_id = np.where(self.book_pivot.index == book_name)[0][0]
            distances, suggestions = self.knn_model.kneighbors(
                self.book_pivot.iloc[book_id, :].values.reshape(1, -1),
                n_neighbors=6
            )
            return [self.book_pivot.index[suggestions[0][i]]
                   for i in range(1, len(suggestions[0]))]
        except Exception as e:
            print(f"Error in rating recommendations: {str(e)}")
            return []
    
    def recommend(self, book_name, num_recommendations=5):
        """
        Get hybrid recommendations combining both image and rating based approaches
        
        Args:
            book_name (str): Name of the book to base recommendations on
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: Recommended book titles
        """
        # Get recommendations from both models
        image_recs = self.get_image_recommendations(book_name)
        rating_recs = self.get_rating_recommendations(book_name)
        
        # Create weighted scores for each unique book
        recommendations = {}
        
        # Score image-based recommendations
        for i, book in enumerate(image_recs):
            if book not in recommendations:
                recommendations[book] = self.image_weight * (1 - i/len(image_recs))
                
        # Score rating-based recommendations
        for i, book in enumerate(rating_recs):
            if book in recommendations:
                recommendations[book] += self.rating_weight * (1 - i/len(rating_recs))
            else:
                recommendations[book] = self.rating_weight * (1 - i/len(rating_recs))
        
        # Sort by score and return top recommendations
        sorted_recs = sorted(recommendations.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
        return [book for book, score in sorted_recs[:num_recommendations]]

recommender = HybridBookRecommender(image_weight=0.3, rating_weight=0.7)
recommend_book = recommender.recommend