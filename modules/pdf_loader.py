from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.schema import Document
import os
from io import BytesIO

class PDFLoader:
    def __init__(self, file_bytes):
        """
        file_bytes: bytes of the PDF upload.
        """
        self.file_bytes = file_bytes

    def load(self):
        """Load the PDF file and return a list of Document objects."""
        pdf_reader = PdfReader(BytesIO(self.file_bytes.getvalue()))
        documents = []
        text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
                
         # Chuyển thành đối tượng Document của LangChain
        documents.append(Document(page_content=raw_text, metadata={"source": "Upload PDF"}))
        return documents
