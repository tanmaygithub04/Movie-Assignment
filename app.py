import streamlit as st
from tmdbv3api import TMDb, Movie
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import ast
import sys
import subprocess

def run_preprocessing():
    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = False


    if not st.session_state.preprocessed:

        with st.status("Fetching latest movie data and generating embeddings..."):
            try:
                result = subprocess.run([sys.executable, "preprocessor.py"], check=True, capture_output=True, text=True)
                st.success("Preprocessing completed successfully!")
                st.session_state.preprocessed = True 
            except subprocess.CalledProcessError as e:
                st.error("Preprocessing Failed!")
                st.stop()

run_preprocessing()

# Load data
df = pd.read_csv('movies_metadata.csv')
df['genres'] = df['genres'].apply(ast.literal_eval)
index = faiss.read_index('movie_index.faiss')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize TMDB
tmdb = TMDb()
tmdb.api_key = st.secrets["tmdb_api_key"] 
tmdb.language = 'en'
movie_api = Movie()

def fetch_movie_data(title):
    print("searching : " +  title)
    search = movie_api.search(title)

    if not search.results:
        print("Not Found  : " +  title)
        return None
    # print(search)
    details = movie_api.details(search[0].id)
    return {
        'title': details.title,
        'overview': details.overview,
        'genres': [g.name for g in details.genres]
    }

# Streamlit UI
st.title('Movie Similarity Analyzer 🎬')
movie_title = st.text_input('Enter a movie title:', placeholder='Enter Movie Name....')
button_clicked = st.button('Find Similar Movies')

if movie_title or button_clicked:
    if movie_title:
        movie_data = fetch_movie_data(movie_title)
        if not movie_data:
            st.error('Movie not found. Check the title or try another.')
        else:
            # Convert user input into embedding 
            query_text = f"Overview: {movie_data['overview']} Genres: {', '.join(movie_data['genres'])}"
            query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
            
            # FAISS search
            scores, indices = index.search(query_embedding, 2)
            # print(indices)
            # print(scores)
            indexed_scores = zip(scores[0], indices[0])
            
            st.subheader('Top Matches:')
            for i, (score, idx) in enumerate(indexed_scores):
                # print(f"FAISS Index: {idx}, Movie Title: {df.iloc[idx]['title']}")
                match = df.iloc[idx]
                # print(match['genres'])
                common_genres = set(movie_data['genres']).intersection(set(match['genres']))
                percentage = (score + 1) * 50 # since I am using cosine similarity it give output in -1,1 so it becomes 0-2 , and to get percentage it is multiplied by 50
                st.write(f"**{i+1}. {match['title']}** ({percentage:.1f}%)")  # this syntax is used to round of the percentage to first decimal places  
                st.markdown(f"Common Genres: {', '.join(common_genres) if common_genres else 'None'}")
                st.divider()
    else:
        st.warning('Please enter a movie title.')

refresh_movies = st.button("Refresh the database and generate new embeddings")
if refresh_movies:
    st.session_state.preprocessed = False
    run_preprocessing()



