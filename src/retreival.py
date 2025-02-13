from utils import State


class Retrieval:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
