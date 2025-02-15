import re
from langchain_community.document_loaders import PyMuPDFLoader


class DataLoader:
    def __init__(self, file_path: str):
        self.loader = PyMuPDFLoader(file_path=file_path)

    def get_docs(self):
        docs = self.loader.load()

        return docs
