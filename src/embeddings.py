from langchain_huggingface import HuggingFaceEmbeddings


class Embeddings:
    def __init__(self, model_name: str):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            print(f"Please, enter a valid model name. {e}")
