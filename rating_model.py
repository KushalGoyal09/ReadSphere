import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load and preprocess books data
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

# Create a pivot table and handle NaN values by filling them with 0
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating').fillna(0)
book_sparse = csr_matrix(book_pivot)

# Train the Nearest Neighbors model
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Function to recommend books based on a given book title
def recommend_book(book_name):
    book_suggestions = []
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        for i in range(1, len(suggestions[0])):
            book_suggestions.append(book_pivot.index[suggestions[0][i]])
    except IndexError:
        book_suggestions = ["Book not found."]
    return book_suggestions
