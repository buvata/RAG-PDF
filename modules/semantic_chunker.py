from langchain.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from config import settings

class SemanticChunkerWrapper:
    def __init__(self):
        """
        Initialize the SemanticChunker with a specified model.
        
        :param model_name: The name of the Hugging Face model to use for embeddings.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    def split_documents(self, documents, threshold=0.95):
        """
        Chunk the input text into semantic segments.
        :param documents: List of documents to be chunked.
        :param threshold: Similarity threshold for chunking.
        :return: List of chunked documents.
        """
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            chunk_size=500,  # Adjust chunk size as needed
            breakpoint_threshold_type="percentile",  # Adjust breakpoint threshold as needed
            breakpoint_threshold_amount=threshold
        )
        chunks = text_splitter.split_documents(documents)

        return chunks