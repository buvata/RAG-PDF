import streamlit as st
from modules.pdf_loader import PDFLoader
from modules.semantic_chunker import SemanticChunkerWrapper
from modules.vector_store import VectorStore
from modules.rag_pipeline import RAGPipeline
from modules.llm_manager import LLMManager

def initialize_rag(file_bytes):
    """
    Initialize the components required for the RAG pipeline.
    """
    st.title("RAG PDF Document Processing")
    try:
        # Load the PDF document
        pdf_loader = PDFLoader(file_bytes)
        documents = pdf_loader.load()
        
        if not documents:
            st.error("No text found in the PDF document.")
            return None
        
        # Chunk the documents semantically
        semantic_chunker = SemanticChunkerWrapper()
        chunks = semantic_chunker.split_documents(documents)
        
        if not chunks:
            st.error("No semantic chunks created from the document.")
            return None
        
        # Initialize vector store and add chunks
        vectorstore = VectorStore()
        vectorstore = vectorstore.create_vector_store(chunks)
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        llm = llm_manager.load_llm(max_tokens=256)        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(vectorstore, llm)
        
        return rag_pipeline
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    
def main():
    st.set_page_config(page_title="RAG PDF Document Processing", layout="wide")
    st.title("RAG PDF Document Processing") 
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        if "file_processed" not in st.session_state or st.session_state.uploaded_file != uploaded_file.name:
            with st.spinner("Đang xử lý file PDF..."):
                rag_pipeline = initialize_rag(uploaded_file)
                if rag_pipeline:
                    st.session_state.rag_pipeline = rag_pipeline
                    st.session_state.uploaded_file = uploaded_file.name
                    st.session_state.file_processed = True
                    st.success("File đã sẵn sàng để tra cứu!")

        else:
            st.info("Vui lòng tải lên một file PDF để bắt đầu.")
            return
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Hỏi điều bạn muốn biết..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            rag_pipeline = st.session_state.rag_pipeline
            result = rag_pipeline.aks_question(prompt)

            response = result["result"]
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()