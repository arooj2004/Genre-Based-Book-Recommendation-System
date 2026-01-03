📚 Genre-Aware Book Recommendation System (Machine Learning Project)

This project is a machine learning–based book recommendation system that suggests relevant books based on genres, textual descriptions, ratings, and popularity. The system is designed as part of a Machine Learning course semester project and demonstrates a complete end-to-end ML pipeline, from data preprocessing to model evaluation and user-friendly deployment.

The project uses a real-world dataset of 10,000 books and applies feature engineering, multiple regression models, and performance evaluation to identify the most effective recommendation approach. A Streamlit web interface is also implemented to allow users to search books by title or genre, view book covers, and receive clean, duplicate-free recommendations.

🔍 Key Features

Genre-based and title-based book recommendations

Cleaned and preprocessed real-world dataset

Feature engineering using TF-IDF, SVD, and multi-label genre encoding

Training and comparison of three ML models:

Linear Regression (baseline)

Random Forest Regressor

Gradient Boosting Regressor (best performing)

Evaluation using RMSE, MAE, and R² score

Interactive Streamlit frontend with:

Book cover images

Sidebar navigation

Card-based UI

No duplicate book results

🧠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

TF-IDF & Truncated SVD

Open Library API (for book covers)

📊 Dataset

Best Books (10k) Multi-Genre Dataset

Source: Kaggle

🎯 Learning Outcomes

Understanding end-to-end machine learning workflows

Practical experience with recommender systems

Model selection and evaluation

Feature engineering for text data

Building ML-powered interactive applications

🚀 How to Run

Train models:

python train_models.py


Launch the web app:

streamlit run app.py
