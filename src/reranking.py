import os
import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

import google.generativeai as genai
from google.generativeai import GenerativeModel

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from utils import State


class DocumentMatch(BaseModel):
    page_number: int = Field(
        ..., description="The page number where the relevant content was found"
    )
    reranking_score: float = Field(
        ..., description="The reranking score of the content to the query"
    )
    content: str = Field(..., description="The extracted text content from the page")


class DocumentMatchList(BaseModel):
    documents: List[DocumentMatch]


class Reranker:
    def __init__(self, model_name: str, model_api_key: str):
        template = """You're an expert document re-ranker where your task is
        re-rank some given documents delimited by the triple backticks below
        according to their relavancy to a user query delimited by angle brackets.

        Make sure to re-rank the documents according to the given query based
        on relevancy where the top document is the most similar one. 

        Your output must be in JSON format:
        {format_instructions}

        Documents:
        ```
        {documents}
        ```

        Query:
        <{query}>

        Reranked Documents:"""

        self.parser = PydanticOutputParser(pydantic_object=DocumentMatchList)
        self.prompt = ChatPromptTemplate.from_template(template)
        self.response_json = None

        assert model_name.lower() in [
            "cohere",
            "gemini",
        ], "`model_name` must be one of the following options: ['cohere', 'gemini']"

        self.model_name = model_name.lower()
        self.prompt_template = ChatPromptTemplate.from_template(template)

        if self.model_name == "cohere":
            if not os.environ.get("COHERE_API_KEY"):
                os.environ["COHERE_API_KEY"] = model_api_key
            self.llm = init_chat_model("command-r-plus", model_provider="cohere")
        else:
            if not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = model_api_key
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.llm = GenerativeModel("gemini-1.5-flash")

    def rerank_documents(self, state: State):
        prompt = self.prompt.invoke(
            {
                "query": state["optimized_query"],
                "documents": state["context"],
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        if self.model_name == "cohere":
            response = self.llm.invoke(prompt)
            response = self.parser.parse(response.content)
        else:
            response = self.llm.generate_content(prompt.to_string())
            response = self.parser.parse(response.text)

        self.response_json = json.loads(response.model_dump_json())["documents"]
        return {"reranked_documents": self.response_json}
