from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def test_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
    except Exception as e:
        print(f"Failed to load vector store: {e}")

if __name__ == "__main__":
    test_vector_store()
