"""
Retrival-Augmented Generation (RAG) - Version: 1.2.0

- Conducted some UI enhancments and optimization of the flow
to have a better user experience.
- Caching the embeddings.
- Implemented a query optimization module for optimize the query for
better retrieval.
- Implemented a document re-ranking strategy for better generation
and savings in the context size.
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
from query_optimization import QueryOptimizer
from retreival import Retrieval
from reranking import Reranker
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

# Load & save the API keys
stored_keys = load_api_keys()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = stored_keys["LangSmith"]
os.environ["GOOGLE_API_KEY"] = stored_keys["Gemini"]
os.environ["COHERE_API_KEY"] = stored_keys["Cohere"]

# Pull the prompt
prompt = hub.pull("rlm/rag-prompt")

# Initialize session state for selected model
if "rag_pipe_trigger" not in st.session_state:
    st.session_state.rag_pipe_trigger = False

if "rag_pipe" not in st.session_state:
    st.session_state.rag_pipe = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_embeddings(model_name="sentence-transformers/all-mpnet-base-v2"):
    embeddings = Embeddings(model_name=model_name)
    return embeddings


def process_pdf(temp_file_path):
    data_loader = DataLoader(temp_file_path)
    docs = data_loader.get_docs()
    return docs


def split_text(_docs, _embeddings):
    text_splitter = TextSplitter(embeddings=_embeddings)
    return text_splitter.split_documents(_docs)


def index_documents(_text_splits, _embeddings):
    index = Index(embeddings=_embeddings)
    index.vectorstore.reset_collection()
    index.add_documents(_text_splits)

    return index


def create_rag_pipeline(model_name, api_key, _index, use_rerank=False):
    query_optimizer = QueryOptimizer(model_name=model_name, model_api_key=api_key)
    retrieval = Retrieval(_index.vectorstore)
    reranker = Reranker(model_name=model_name, model_api_key=api_key)
    generator = Generator(prompt=prompt, model_name=model_name, model_api_key=api_key)

    return RagPipeline(
        prompt, query_optimizer, retrieval, reranker, generator, use_rerank
    )


with st.sidebar:
    st.subheader("1. Choose a Model")
    model_name = st.selectbox(
        "Choose a model",
        ["Cohere", "Gemini"],
    )

    if model_name == "Cohere":
        env_var = "COHERE_API_KEY"
    else:
        env_var = "GOOGLE_API_KEY"

    # Display the toggle to show the retrieved context and reranking
    st.subheader("2. Configurations")
    st.session_state.show_context = st.toggle(
        label="Show Retrieved Context",
        value=False,
        help="Identify whether to view the retrieved documents or not.",
    )

    st.session_state.rerank = st.toggle(
        label="Re-rank Context",
        value=False,
        help="""Identify whether to re-rank the retrieved documents or not. Note: this
        might take longer but ensures that only relvant context is used while generation, 
        which may lead to better responses sometimes.""",
    )

    st.subheader("3. Upload your PDF")
    uploaded_file = st.file_uploader(label="Upload your PDF", type="pdf")

    if st.button("Submit"):
        if uploaded_file:
            with st.spinner("Processing your file..."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                embeddings = load_embeddings()
                docs = process_pdf(temp_file_path)
                text_splits = split_text(docs, embeddings)
                index = index_documents(text_splits, embeddings)
                st.session_state.index = index

            st.success("Indexing complete successfully!")
            st.session_state.rag_pipe = create_rag_pipeline(
                model_name, os.environ[env_var], index, st.session_state.rerank
            )
            st.session_state.rag_pipe_trigger = True

        else:
            st.error("Please, upload a PDF document")


pdf_preview_area, chat_area = st.columns(2)

# Pdf preview container
with pdf_preview_area:
    st.subheader("PDF Preview", anchor=False)

    if uploaded_file:
        with st.container(border=True):
            binary_data = uploaded_file.getvalue()
            pdf_viewer(
                input=binary_data,
                height=600,
                resolution_boost=2,
                pages_vertical_spacing=1,
                render_text=True,
            )


# Chat area container
with chat_area:
    st.subheader("Chat Interface", anchor=False)
    if st.session_state.rag_pipe_trigger:
        # Re-create a new RAG pipeline when the user trigger the re-rank toggle
        if st.session_state.rag_pipe.use_rerank != st.session_state.rerank:
            st.session_state.rag_pipe = create_rag_pipeline(
                model_name,
                os.environ[env_var],
                st.session_state.index,
                st.session_state.rerank,
            )

        chat = st.container(height=580, border=False)
        question = st.chat_input("Ask anything related to your document...")
        hello_message = "Hello! 👋 I'm here to help you find answers from your PDFs. How can I assist you today?"

        if len(st.session_state.messages) == 0:
            st.session_state.messages.append(
                {"role": "assistant", "content": hello_message}
            )

        with chat:
            # Stream the hello message
            if len(st.session_state.messages) < 2:
                with st.chat_message("assistant", avatar=ai_avatar):
                    st.write_stream(response_generator(answer=hello_message))

            # Display chat messages from history on app rerun
            for message in st.session_state.messages[1:]:
                avatar = mohammed_avatar if message["role"] == "user" else ai_avatar
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])

            if question:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": question})

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
                            st.markdown("**Retrieved documents:**")
                            st.write(context)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": context}
                            )

                    # Display the reranking results
                    if st.session_state.rerank == True:
                        st.markdown("**Reranking results:**")
                        reranking_results = (
                            st.session_state.rag_pipe.reranker.response_json
                        )

                        st.write(reranking_results)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reranking_results}
                        )

                        st.markdown("**The passed context to the model:**")
                        st.write(reranking_results[:2])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reranking_results[0]}
                        )

                    # Stream the response
                    response = st.write_stream(response_generator(answer=answer))

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
