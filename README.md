# 🎬 Dynamic Movie Similarity Analyzer

A system that recommends movies based on storyline, genre, and summary similarity using semantic embeddings utilising SentenceTransformer which were indexed using FAISS for efficient similarity search.

## Deployed Streamlit Cloud Link :- https://movie-assignment-lqzekaykk3rnewmby3mkkf.streamlit.app/

## How to run locally 
 ```sh
    git clone <https://github.com/tanmaygithub04/Movie-Assignment>
    cd Movie-Assignment
    pip install requirements.txt
    streamlit run app.py
   ```



## ✨ Features
- **Dynamic Movie Processing** Fetch unseen movies via TMDB API
- **Generates text embeddings** with Sentence Transformers.
- **Indexes these embeddings** using FAISS for fast similarity search.
- Serves a user interface via **Streamlit**, allowing users to type in a movie title and get top matching movies and also refresh the database to get new embeddings.



##  Workflow
### Data Pipeline
1. **Data Preprocessing**:
   - Fetch popular movies from TMDB API and select random 20 from them
   - Store these in CSV names movies_metadata.csv with columns: `title`, `overview`, `genres`, `combined_text` 
   - Create embeddings of combined values : `title`, `overview`, `genres` and that string is stored as `combined_text` 
   - Embeddings of this `combined_text` is created using SentenceTransformer and indexed these uisng ``index.add(embeddings)``

2. **Embedding Generation**:
   - Use SentenceTransformer (similar/better than BERT for recommendation systems) model , seen in this article :- https://medium.com/@mroko001/transformers-in-nlp-bert-and-sentence-transformers-3faab61918ea#:~:text=BERT%20excels%20at%20word%2Dlevelachieving%20human%2Dlevel%20language%20understanding.
   - Combine overview + genres into single text
   - We can use this in case the database is very large Ref is this article :- https://www.pinecone.io/learn/series/faiss/vector-indexes/#Locality-Sensitive-Hashing
index.add(embeddings)

3. **Details on FAISS Indexing**:
   - Learned Vector embedding and FAISS indexing for better operation speed / similarity matching 
   - I chose FLAT since it gives better quality but at cost of lower speeds , but since we dont have a huge database right now it is fine
   - Normalized IP embeddings for cosine similarity
   - Was having problem in writing/finding sources for writing some syntax so took help from this article :- https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772
   - Algorithm used is KNN to find the closest matching movies 

4. **Details on Deployment and Frontend**:
   - Frontend in implemented using Streamlit for easier deployment
   - I was initially thinking of creating a pipleline using github actions but that was not suitable here since I decided  we need to preprocess everytime the page loads
   - 
  



  

