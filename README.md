# 🎬 Dynamic Movie Similarity Analyzer

A system that recommends movies based on storyline, genre, and summary similarity using semantic embeddings utilising SentenceTransformer which were indexed using FAISS for efficient similarity search.


## ✨ Features
- **Dynamic Movie Processing** Fetch unseen movies via TMDB API
- **Generates text embeddings** with Sentence Transformers.
- **Indexes these embeddings** using FAISS for fast similarity search.
- Serves a user interface via **Streamlit**, allowing users to type in a movie title and get top matching movies and also refresh the database to get new embeddings.



## 🛠️ Workflow
### Data Pipeline
1. **Movie Collection**: 
   - Fetch 20 popular movies from TMDB API
   - Store in SQLite with columns: `title`, `overview`, `genres`
2. **Embedding Generation**:
   - Use SentenceTransformer (similar/better than BERT for recommendation systems) model , seen in this article :- https://medium.com/@mroko001/transformers-in-nlp-bert-and-sentence-transformers-3faab61918ea#:~:text=BERT%20excels%20at%20word%2Dlevel,achieving%20human%2Dlevel%20language%20understanding.
   - Combine overview + genres into single text
   - I chose FLAT since it gives better quality at lower speeds and we dont have a huge database right now 
   - We can use this in case the database is very large Ref is this article :- https://www.pinecone.io/learn/series/faiss/vector-indexes/#Locality-Sensitive-Hashing
index.add(embeddings)
3. **Details on Embedding**:
   - Learned Vector embedding and FAISS indexing for better operation speed / similarity matching 
   - Normalized IP embeddings for cosine similarity
   - Was having problem in writing/finding sources for writing some syntax so took help from this article :- https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772
   - Algorithm used is KNN to find the closest matching movies 

  

