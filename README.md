

#Watch out Video-->

https://github.com/user-attachments/assets/e467f504-f18c-410e-bfa1-a8748ddf5088



# Recommendation Systems

This repository contains implementations of Content-Based Filtering, Collaborative Filtering, and Hybrid Recommendation Systems. These methods are commonly used in applications such as e-commerce, streaming services, and social media platforms to suggest relevant items to users.

## Table of Contents

- [Overview](#overview)
- [Recommendation Techniques](#recommendation-techniques)
  - [Content-Based Filtering](#content-based-filtering)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Hybrid Recommendation System](#hybrid-recommendation-system)

## Overview

A recommendation system suggests items to users based on their preferences, past behavior, and similarities with other users or items. This project explores three main techniques:

- **Content-Based Filtering**: Recommends items similar to what a user has interacted with.
- **Collaborative Filtering**: Suggests items based on interactions of users with similar interests.
- **Hybrid Model**: Combines content-based and collaborative filtering for improved accuracy.

## Recommendation Techniques

### Content-Based Filtering

- Uses item attributes (e.g., genre, keywords, descriptions) to recommend similar items.
- Measures similarity using techniques like **TF-IDF, cosine similarity, and word embeddings**.
- Works well when sufficient metadata is available for items.

### Collaborative Filtering

- Uses past user interactions to find similarities among users or items.
- Two types:
  - **User-Based Collaborative Filtering**: Finds users with similar preferences.
  - **Item-Based Collaborative Filtering**: Recommends items that are frequently liked together.
- Common techniques include **KNN, Matrix Factorization (SVD, ALS), and Deep Learning models**.

### Hybrid Recommendation System

- Combines both content-based and collaborative filtering.
- Helps mitigate cold-start problems and improves recommendation diversity.
- **Netflix, Amazon, and Spotify** use hybrid models for better personalization.




# Movie Recommender System

## 🎬 Overview
This Movie Recommender System suggests movies based on user preferences using text analysis. It analyzes movie details such as genres, keywords, cast, and crew to determine similarities between films. By leveraging NLP techniques and machine learning, it provides accurate recommendations efficiently.

## 🚀 Features
- **Content-Based Filtering**: Uses movie metadata to find similar films.
- **Text Preprocessing**: Implements **PorterStemmer** from NLTK to normalize text.
- **Vectorization**: Utilizes **CountVectorizer** to convert text data into numerical format.
- **Similarity Calculation**: Uses **cosine similarity** to measure closeness between movies.
- **Optimized Performance**: Speeds up operations using **Pandas**.
- **Interactive UI**: Built with **Streamlit** for user-friendly movie recommendations.
- **Efficient Data Storage**: Saves processed data and similarity matrix using **pickle** for quick retrieval.

## 🛠️ Technologies Used
- **Python** (pandas, numpy, nltk, scikit-learn, streamlit, pickle)
- **Natural Language Processing (NLP)** (NLTK’s PorterStemmer)
- **Machine Learning** (CountVectorizer, Cosine Similarity)

## 🔧 Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/shree_sriv12/movie-recommender-system.git
Sorry: you got to install  the credits.csv file from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata🥲
cd movie-recommender-system
```


### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py or python3 -m streamlit run app.py(runs on my system ><)
```

## 📌 How It Works
1. Loads movie dataset and processes metadata.
2. Cleans and stems text data using **PorterStemmer**.
3. Converts text into a **bag-of-words model** using **CountVectorizer**.
4. Computes **cosine similarity** between movies.
5. Stores processed data using **pickle** for fast access.
6. Provides movie recommendations based on user input.

## 🎯 Usage Example
1. Open the Streamlit app.
2. Enter a movie title in the search box.
3. The system suggests similar movies instantly!

## 📁 File Structure
```
📂 movie-recommender
├── app.py              # Streamlit app
├── model.py            # Recommendation system logic
├── movies.pkl          # Preprocessed movie data
├── similarity.pkl      # Cosine similarity matrix
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```


🌟 **Enjoy your movie recommendations!** 🎬🍿



