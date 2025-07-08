from langchain_mongodb import MongoDBAtlasVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from config import settings

class VectorStore:
    def __init__(self):
        """
        Initialize the VectorStore with MongoDB connection and embeddings.
        """
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.DB_NAME]
        self.collection = self.db[settings.COLLECTION_NAME]

    def create_vector_store(self, chunks):
        """
        Create a vector store using MongoDB Atlas.
        :return: An instance of MongoDBAtlasVectorStore.
        """
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        vectorstore = MongoDBAtlasVectorStore.from_documents(
            collection=self.collection,
            embedding=embeddings,
            index_name=settings.VECTOR_INDEX_NAME,
            documents=chunks,  # Documents can be provided later
        )
        return vectorstore
