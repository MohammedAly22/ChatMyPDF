import os
from utils import State

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

import google.generativeai as genai
from google.generativeai import GenerativeModel


class Generator:
    def __init__(self, prompt: ChatPromptTemplate, model_name: str, model_api_key: str):
        assert model_name.lower() in [
            "cohere",
            "gemini",
        ], "`model_name` must be one of the following options: ['cohere', 'gemini']"

        self.model_name = model_name
        self.prompt = prompt

        if model_name == "cohere":
            if not os.environ.get("COHERE_API_KEY"):
                os.environ["COHERE_API_KEY"] = model_api_key
            self.llm = init_chat_model("command-r-plus", model_provider="cohere")
        else:
            if not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = model_api_key
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.llm = GenerativeModel("gemini-1.5-flash")

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        if state.get("reranked_documents"):
            messages = self.prompt.invoke(
                {
                    "question": state["optimized_query"],
                    "context": state["reranked_documents"][:2],
                }
            )
        else:
            messages = self.prompt.invoke(
                {
                    "question": state["optimized_query"],
                    "context": docs_content,
                }
            )

        if self.model_name == "cohere":
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        else:
            response = self.llm.generate_content(str(messages))
            return {"answer": response.text}
