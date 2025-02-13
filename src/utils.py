import os
import json
import time

from typing_extensions import TypedDict
from typing import List

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
    context: List[Document]
    answer: str


def response_generator(answer: str):
    for word in answer.split():
        yield word + " "
        time.sleep(0.005)


def build_and_compile_graph(retrieve, generate):
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
