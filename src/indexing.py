from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from embeddings import Embeddings


class Index:
    def __init__(self, embeddings: Embeddings):
        self.vectorstore = Chroma(
            collection_name="rag-app",
            embedding_function=embeddings.embeddings,
            persist_directory="./data/chroma_rag_db",
        )

    def add_documents(self, docs: List[Document]):
        _ = self.vectorstore.add_documents(documents=docs)
