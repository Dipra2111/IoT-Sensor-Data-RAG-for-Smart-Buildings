from sentence_transformers import SentenceTransformer
if __name__ == "__main__":
    print("Downloading 'all-MiniLM-L6-v2'...")
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model cached.")
