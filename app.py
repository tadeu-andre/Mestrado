import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# Configuração Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "turismo-web"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeStore.from_existing_index(index_name=index_name, embedding=embeddings)

retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template(
    """Você é um guia turístico inteligente. Responda à pergunta com base nos documentos:
    Pergunta: {input}
    Contexto: {context}
    Resposta:"""
)
combine_docs_chain = create_stuff_documents_chain(llm=None, prompt=prompt)  # Substituir LLM real depois
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Interface
st.title("Assistente de Turismo RAG")
user_question = st.text_input("Digite sua pergunta sobre turismo:")
if user_question:
    response = rag_chain.invoke({"input": user_question})
    st.write(response)
