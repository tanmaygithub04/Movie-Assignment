import pandas as pd
from tmdbv3api import TMDb, Movie
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random

# Initialize TMDB
tmdb = TMDb()
tmdb.api_key = '1d7c23ce27c4f86a2a66312a4b34d724' 
tmdb.language = 'en'

movie_obj = Movie()

# Fetch popular movies list 
popular_movies = movie_obj.popular()

top_popular_movies = []
movie_set = set()
# Selcet first 20 movies if we want to select random movies that can also be done 
while(len(top_popular_movies) < 20) :
    rand_idx = random.randint(0, len(popular_movies)-1)
    if(popular_movies[rand_idx].id not in movie_set):
        movie_set.add(popular_movies[rand_idx].id)
        top_popular_movies.append(popular_movies[rand_idx])

    

# Storing some data from the list 
movies_data = []
for movie in top_popular_movies:
    details = movie_obj.details(movie.id)
    genres = [genre.name for genre in details.genres]
    movies_data.append({
        'title': details.title,
        'overview': details.overview,
        'genres': genres
    })

# Create DataFrame and combine text
df = pd.DataFrame(movies_data)
df['combined_text'] = df.apply(lambda x: f"Overview: {x['overview']} Genres: {', '.join(x['genres'])}", axis=1)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined_text'], normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

# Build and save FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # I chose FLAT since it gives better quality at lower speeds and we dont have a huge database right now 
# index = faiss.IndexLSH(embeddings.shape[1], embeddings.shape[1]*32)  # We can use this in case the database is very large Ref is this article :- https://www.pinecone.io/learn/series/faiss/vector-indexes/#Locality-Sensitive-Hashing
index.add(embeddings)
faiss.write_index(index, 'movie_index.faiss')

# Save metadata
df.to_csv('movies_metadata.csv', index=False)
