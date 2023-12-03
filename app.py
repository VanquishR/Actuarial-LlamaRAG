import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings, CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(vectorstore):
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_vectorstore(text_chunks):
    # HF Version
    # embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    embeddings = CohereEmbeddings(cohere_api_key="awrpR5LiW5iGICpUmXwhiOwbP3cFn8jkBHEJVLbm")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_pdf_txt(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    # Check if the response contains the chat history
    if response and 'chat_history' in response:
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        # Handle the case where chat history is not present in the response
        st.write("Error: No chat history found in the response.")



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs!",
                       page_icon=":books")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PDFs! :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    # Assigning the input and button widgets to variables
    user_question = st.text_input("Ask a Question!:")
    if user_question:
        handle_userinput(user_question)

    # process_button = st.button("Process")
    st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    # Using the assigned variables in your logic (replace this with your actual logic)
    # if process_button:
    #    st.write(f"Question: {question}")
    #   st.write(f"Uploaded File: {uploaded_file}")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Put your files here and click on Process!",
                                    accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing!"):
                # First, get the PDF Text
                raw_text = get_pdf_txt(pdf_docs)

                # Then, convert text into manageable chunks.
                text_chunks = get_text_chunks(raw_text)

                st.write(text_chunks)

                # Now, load these into a vector database.
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
