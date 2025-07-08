from langchain.chains import RetrievalQA

class RAGPipeline:
    def __init__(self, vectorstore, llm):
        """
        Initialize the RAG pipeline with a vector store and a language model.
        
        :param vector_store: An instance of the vector store to retrieve documents.
        :param llm: An instance of the language model for question answering.
        """
        self.vectorstore = vectorstore
        self.llm = llm

    def create_chain(self):
        """
        Create a RetrievalQA chain using the provided vector store and language model.
        
        :return: An instance of RetrievalQA chain.
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            return_source_documents=True,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    def ask_question(self, question):
        """
        Ask a question using the RAG pipeline and return the answer along with source documents.
        
        :param question: The question to ask.
        :return: A tuple containing the answer and the source documents.
        """
        chain = self.create_chain()
        result = chain({"query": question})
        return result['result'], result['source_documents']
    