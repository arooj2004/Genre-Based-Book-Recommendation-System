import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from difflib import get_close_matches

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.book-card {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.book-title {
    font-size: 20px;
    font-weight: 700;
}
.book-author {
    color: #555;
    font-size: 14px;
}
.book-meta {
    font-size: 14px;
    margin-top: 6px;
}
.genre-badge {
    display: inline-block;
    background-color: #2f2f30;
    color: #eef2ff;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 6px;
    margin-top: 6px;
}
.hero {
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA & MODELS ----------------
df = pd.read_pickle("models/books.pkl")
mlb = pickle.load(open("models/mlb.pkl", "rb"))

genre_matrix = mlb.transform(df["Genres"])

# ---------------- COVER IMAGE FUNCTION ----------------
@st.cache_data
def fetch_cover(title):
    try:
        url = f"https://openlibrary.org/search.json?title={title}"
        data = requests.get(url).json()
        cover_id = data["docs"][0].get("cover_i")
        if cover_id:
            return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
    except:
        pass
    return "https://via.placeholder.com/160x240.png?text=No+Cover"

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_books(indices, top_n=10):
    seen = set()
    results = []

    for i in indices:
        title = df.iloc[i]["Book"]
        if title not in seen:
            seen.add(title)
            results.append(i)
        if len(results) == top_n:
            break

    return df.iloc[results]

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>📚 Book Recommendation System</h1>
    <p>Machine Learning–based recommendations using genres, popularity, and content</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔎 Search Options")
search_mode = st.sidebar.radio(
    "Choose Search Type",
    ["Search by Title", "Search by Genre"]
)

top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# ---------------- TITLE SEARCH ----------------
if search_mode == "Search by Title":
    st.subheader("🔍 Find Similar Books")

    query = st.text_input("Enter a book title")

    if query:
        matches = get_close_matches(query, df["Book"].tolist(), n=10, cutoff=0.4)

        if matches:
            selected = st.selectbox("Select a book", matches)

            if st.button("Recommend"):
                idx = df[df["Book"] == selected].index[0]
                scores = genre_matrix @ genre_matrix[idx]
                indices = np.argsort(scores)[::-1]

                results = recommend_books(indices, top_n)

                for _, row in results.iterrows():
                    with st.container():
                        st.markdown('<div class="book-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([1, 4])

                        with col1:
                            st.image(fetch_cover(row["Book"]), width=140)

                        with col2:
                            st.markdown(f"<div class='book-title'>{row['Book']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='book-author'>by {row['Author']}</div>", unsafe_allow_html=True)

                            st.markdown(
                                f"<div class='book-meta'>⭐ {row['Avg_Rating']} | 👥 {int(row['Num_Ratings']):,} ratings</div>",
                                unsafe_allow_html=True
                            )

                            for g in sorted(set(row["Genres"])):
                                st.markdown(f"<span class='genre-badge'>{g}</span>", unsafe_allow_html=True)

                            st.markdown(f"[🔗 View on Goodreads]({row['URL']})")

                        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.subheader("🎭 Search Books by Genre")

    genres = sorted(mlb.classes_)
    selected_genres = st.multiselect("Select one or more genres", genres)

    if st.button("Recommend Books"):
        if selected_genres:
            genre_idx = [genres.index(g) for g in selected_genres]
            scores = genre_matrix[:, genre_idx].sum(axis=1)
            indices = np.argsort(scores)[::-1]

            results = recommend_books(indices, top_n)

            for _, row in results.iterrows():
                st.markdown('<div class="book-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 4])

                with col1:
                    st.image(fetch_cover(row["Book"]), width=140)

                with col2:
                    st.markdown(f"<div class='book-title'>{row['Book']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-author'>by {row['Author']}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='book-meta'>⭐ {row['Avg_Rating']} | 👥 {int(row['Num_Ratings']):,}</div>",
                        unsafe_allow_html=True
                    )

                    for g in sorted(set(row["Genres"])):
                        st.markdown(f"<span class='genre-badge'>{g}</span>", unsafe_allow_html=True)

                    st.markdown(f"[🔗 View on Goodreads]({row['URL']})")

                st.markdown("</div>", unsafe_allow_html=True)