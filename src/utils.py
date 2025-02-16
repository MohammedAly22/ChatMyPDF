import os
import json
import time

from typing_extensions import TypedDict
from typing import List, Tuple, Dict

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph


API_KEYS_FILE = "./api_keys.json"


def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {}


class State(TypedDict):
    question: str
    optimized_query: str
    context: List[Document]
    reranked_documents: Dict
    answer: str


def response_generator(answer: str):
    for char in answer:
        yield char
        time.sleep(0.002)


def prepare_context(retrieved_docs: List[Tuple[Document, float]]) -> List[str]:
    context = []
    for doc, score in retrieved_docs:
        page_number = doc.metadata["page"]
        content = doc.page_content

        doc_object = {
            "Page Number": page_number,
            "Similarity Score": f"{score:.4f}",
            "Content": content,
        }

        context.append(doc_object)

    return context


def build_and_compile_graph(optimize_query, retrieve, rerank_documents, generate):
    if rerank_documents:
        graph_builder = StateGraph(State).add_sequence(
            [optimize_query, retrieve, rerank_documents, generate]
        )
    else:
        graph_builder = StateGraph(State).add_sequence(
            [optimize_query, retrieve, generate]
        )

    graph_builder.add_edge(START, "optimize_query")
    graph = graph_builder.compile()

    return graph
