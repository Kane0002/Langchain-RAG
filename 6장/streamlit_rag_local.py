import os
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Set the OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = "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"

@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Create a vector store from the document chunks
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small'))
    return vectorstore

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

# Initialize the LangChain components
@st.cache_resource
def chaining():
    file_path = r"C:\Users\gram\Downloads\대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = create_vector_store(pages)
    retriever = vectorstore.as_retriever()

    # Define the answer question prompt
    qa_system_prompt = """
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    Please answer in Korean and use respectful language.\
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o")
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit UI
st.header("헌법 Q&A 챗봇 💬 📚")
rag_chain = chaining()


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt_message)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
            