"""
Retrival-Augmented Generation (RAG) - Version: 1.1.0

- Using `ChromaDB` vector database instead of `InMemoryVectorStore`.
- Using `PyMuPDFLoader` for data loading instead of `PyPDFLoader`.
- Using `SemanticChuncker` for a more advanced chunking strategy. 
- Using `similarity_search_with_relevance_scores` function for retrieval.
- Adding the functionality of showing the retrieved context.
"""

import os
import tempfile

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from langchain import hub

from ingestion import DataLoader
from chunking import TextSplitter
from embeddings import Embeddings
from indexing import Index
from retreival import Retrieval
from generation import Generator
from rag_pipeline import RagPipeline

from utils import response_generator, load_api_keys, prepare_context

# Set page layout
st.set_page_config(layout="wide")


# Set custom CSS styling
with open("src/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load message avatars
mohammed_avatar = "docs/Mohammed_avatar.png"
ai_avatar = "docs/assistant_avatar.jpg"

# Load existing API keys
stored_keys = load_api_keys()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = stored_keys["LangSmith"]
os.environ["GOOGLE_API_KEY"] = stored_keys["Gemini"]
os.environ["COHERE_API_KEY"] = stored_keys["Cohere"]

# Pull the prompt
prompt = hub.pull("rlm/rag-prompt")


# Initialize session state for selected model and API key
if "rag_pipe_trigger" not in st.session_state:
    st.session_state.rag_pipe_trigger = False

if "rag_pipe" not in st.session_state:
    st.session_state.rag_pipe = None


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("1. Choose a Model")
    model_name = st.selectbox(
        "Choose a model",
        ["Cohere", "Gemini"],
    )

    if model_name == "Cohere":
        cohere_api_key = stored_keys[model_name]
        os.environ["COHERE_API_KEY"] = cohere_api_key
        env_var = "COHERE_API_KEY"
    else:
        google_api_key = stored_keys[model_name]
        os.environ["GOOGLE_API_KEY"] = google_api_key
        env_var = "GOOGLE_API_KEY"

    # Display the toggle to show the retrieved context
    st.subheader("2. Show Context")
    st.session_state.show_context = st.toggle(
        label="Show Retrieved Context",
        value=False,
        help="Identify whether to view the retrieved documents or not.",
    )

    st.subheader("3. Upload your PDF")
    uploaded_file = st.file_uploader(label="Upload your PDF", type="PDF")

    # 1. Download the embeddings
    embeddings = Embeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if st.button("Submit"):
        if uploaded_file is not None:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            with st.spinner("Reading your file..."):
                # 2. Load the PDF file
                data_loader = DataLoader(temp_file_path)
                docs = data_loader.get_docs()

                # 3. Split the PDF file
                text_splitter = TextSplitter(embeddings=embeddings)
                text_splits = text_splitter.split_documents(docs)

            # 4. Index the `text_splits` into the `ChromaDB`
            progress_bar = st.progress(0)
            status_text = st.empty()

            index = Index(embeddings=embeddings)
            status_text.markdown("Indexing Documents...")

            num_docs = len(text_splits)
            for i, doc in enumerate(text_splits):
                index.add_documents([doc])
                # Update progress bar
                progress = int(((i + 1) / num_docs) * 100)
                progress_bar.progress(progress)
                status_text.markdown(f"Indexing document {i + 1} of {num_docs}...")

            progress_bar.progress(100)
            st.success("Indexing has been done successfully!")

            # 5. Create the retrieval
            retrieval = Retrieval(index.vectorstore)

            # 6. Create the generator
            generator = Generator(
                prompt=prompt,
                model_name=model_name,
                model_api_key=os.environ[env_var],
            )

            # 7. Create the RAG pipeline
            rag_pipe = RagPipeline(prompt, retrieval, generator)
            st.session_state.rag_pipe = rag_pipe
            st.session_state.rag_pipe_trigger = True

        else:
            st.error("Please, upload a PDF document")


pdf_preview_area, chat_area = st.columns(2)

with pdf_preview_area:
    st.subheader("PDF Preview", anchor=False)

    if uploaded_file:
        with st.container(border=True):
            if uploaded_file:
                binary_data = uploaded_file.getvalue()
                pdf_viewer(
                    input=binary_data,
                    height=600,
                    resolution_boost=2,
                    pages_vertical_spacing=1,
                )


with chat_area:
    st.subheader("Chat Interface", anchor=False)
    if uploaded_file:
        if st.session_state.rag_pipe_trigger:
            chat = st.container(height=580, border=False)
            question = st.chat_input("Ask anything related to your PDF...")

            with chat:
                # Display hello message
                if len(st.session_state.messages) == 0:
                    hello_message = "Hello! 👋 I'm here to help you find answers from your PDFs. How can I assist you today?"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": hello_message}
                    )
                    with st.chat_message("assistant", avatar=ai_avatar):
                        response = st.write_stream(
                            response_generator(answer=hello_message)
                        )

                # Display chat messages from history on app rerun
                for message in st.session_state.messages[1:]:
                    avatar = mohammed_avatar if message["role"] == "user" else ai_avatar
                    with st.chat_message(message["role"], avatar=avatar):
                        st.write(message["content"])

                if question:
                    # Add user message to chat history
                    st.session_state.messages.append(
                        {"role": "user", "content": question}
                    )
                    # Display user message in chat message container
                    with st.chat_message("user", avatar=mohammed_avatar):
                        st.markdown(question)

                    # Display assistant response in chat message container
                    with st.chat_message("assistant", avatar=ai_avatar):
                        with st.spinner("Answering your question..."):
                            answer = st.session_state.rag_pipe.get_answer(question)

                        # Display the retrieved context
                        if st.session_state.show_context:
                            retrieved_docs = (
                                st.session_state.rag_pipe.retrieval.retrieved_docs
                            )
                            context = prepare_context(retrieved_docs)
                            if context:
                                st.write(context)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": context}
                                )

                    with st.chat_message("assistant", avatar=ai_avatar):
                        response = st.write_stream(response_generator(answer=answer))

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
