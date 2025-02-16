from langchain_core.prompts import ChatPromptTemplate

from retreival import Retrieval
from generation import Generator
from query_optimization import QueryOptimizer
from reranking import Reranker

from utils import build_and_compile_graph


class RagPipeline:
    def __init__(
        self,
        prompt: ChatPromptTemplate,
        query_optimizer: QueryOptimizer,
        retrieval: Retrieval,
        reranker: Reranker,
        generator: Generator,
        use_rerank: bool = False,
    ):
        self.prompt = prompt
        self.query_optimizer = query_optimizer
        self.retrieval = retrieval
        self.reranker = reranker
        self.generator = generator
        self.use_rerank = use_rerank

        if use_rerank:
            self.graph = build_and_compile_graph(
                optimize_query=query_optimizer.optimize_query,
                retrieve=retrieval.retrieve,
                rerank_documents=reranker.rerank_documents,
                generate=generator.generate,
            )
        else:
            self.graph = build_and_compile_graph(
                optimize_query=query_optimizer.optimize_query,
                retrieve=retrieval.retrieve,
                rerank_documents=None,
                generate=generator.generate,
            )

    def get_answer(self, question: str) -> str:
        self.response = self.graph.invoke({"question": question})

        return self.response["answer"]
