from langchain_core.prompts import ChatPromptTemplate


from retreival import Retrieval
from generation import Generator


from utils import build_and_compile_graph


class RagPipeline:
    def __init__(
        self, prompt: ChatPromptTemplate, retrieval: Retrieval, generator: Generator
    ):
        self.prompt = prompt
        self.retrieval = retrieval
        self.generator = generator
        self.graph = build_and_compile_graph(
            retrieve=retrieval.retrieve, generate=generator.generate
        )

    def get_answer(self, question: str) -> str:
        self.response = self.graph.invoke({"question": question})

        return self.response["answer"]
