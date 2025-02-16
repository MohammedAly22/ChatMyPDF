from utils import State


class Retrieval:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, state: State):
        self.retrieved_docs = self.vector_store.similarity_search_with_relevance_scores(
            state["optimized_query"]
        )
        return {"context": [doc for doc, _ in self.retrieved_docs]}
