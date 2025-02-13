from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from embeddings import Embeddings


class Index:
    def __init__(self, embeddings: Embeddings):
        self.vectorstore = InMemoryVectorStore(embedding=embeddings.embeddings)

    def add_documents(self, docs: List[Document]):
        _ = self.vectorstore.add_documents(documents=docs)
