from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGManager:
    """Handles vector search over meeting transcripts for quick intelligence retrieval."""
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def build_index(self, transcript_text):
        texts = self.text_splitter.split_text(transcript_text)
        vector_db = FAISS.from_texts(texts, self.embeddings)
        return vector_db

    def query_context(self, vector_db, query):
        docs = vector_db.similarity_search(query, k=3)
        return " ".join([d.page_content for d in docs])
