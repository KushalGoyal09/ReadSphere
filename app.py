from flask import Flask
from flask import render_template, request, url_for, redirect
import pandas as pd
import numpy as np
from hybrid import final_rating, recommend_book, books

book_image = pd.read_csv('books.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
book_image.rename(columns={'Book-Title':'title', 'Image-URL-M':'image'}, inplace=True)

app = Flask(__name__)

def get_column_list(csv_table, column):
    new_table = csv_table.drop_duplicates(column)
    column_list = new_table[column].tolist()
    return column_list
    
    
def get_specified_column(table_name, column_cond, cond, column_to_find):
    return table_name[table_name[column_cond]==cond][column_to_find].tolist()[0]

book_name = []

@app.route('/', methods=['GET', 'POST'])
def home_page():
    
    books = get_column_list(final_rating, 'title')
    
    if request.method == "POST":
        book = request.form.get('book_name')
        book_name.append(book)
        return redirect(url_for('recommend_page'))
    
    return render_template('home_page.html', books=books, length=len(books))


@app.route('/recommend')
def recommend_page():
    
    book_image = pd.read_csv('books.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
    book_image.rename(columns={'Book-Title':'title', 'Image-URL-M':'image'}, inplace=True)
    
    recommend_book_list = recommend_book(book_name[-1])
    print("RECOMANDED BOOK LIST: ", recommend_book_list)
    authors = []
    image_url = []
    
    for i in range(5):
       print("RECOMMENDED BOOK LIST: ",i ,recommend_book_list[i])
       authors.append(get_specified_column(books, 'title', recommend_book_list[i], 'author'))
       image_url.append(get_specified_column(book_image, 'title', recommend_book_list[i], 'image'))
           
    return render_template('recommendation_page.html', book_name=book_name[-1], recommend_book_list=recommend_book_list, authors=authors, image_url=image_url)

