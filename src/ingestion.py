from langchain_community.document_loaders import PyPDFLoader


class DataLoader:
    def __init__(self, file_path: str):
        self.loader = PyPDFLoader(file_path=file_path)

    def get_docs(self):
        docs = self.loader.load()
        return docs
