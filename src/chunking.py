from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from embeddings import Embeddings


class TextSplitter:
    def __init__(self, embeddings: Embeddings):
        self.text_splitter = SemanticChunker(
            embeddings=embeddings.embeddings, breakpoint_threshold_type="percentile"
        )

    def split_documents(self, docs: List[Document]):
        text_splits = self.text_splitter.split_documents(docs)
        return text_splits
