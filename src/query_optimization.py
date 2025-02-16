import os

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

import google.generativeai as genai
from google.generativeai import GenerativeModel

from utils import State


class QueryOptimizer:
    def __init__(self, model_name: str, model_api_key: str):
        template = """You're an excellent query optimizer, your task is to
        optimize the passed query below in a way that it becomes more clearer, 
        rich, and complete. This will play a crucial role in the retrieval 
        phase later on. So, explain any concepts in the query that would enhance
        the retrieval module to get the desired output. 

        Don't add unnecessary content in the new optimized query, and consider
        that your response will be used for the retrieval phase to retrieve the
        most similar documents to that query.

        Make sure that your response is 3 sentences at maximum and to stick with
        the format in the example down below.

        Example:
        Query: 
        Provide to me the Table of Contents

        Optimized query: Provide to me the Table of Contents; which is a numbered
        list containing the different sections of the contents of this document 
        ----

        Query: {query}
        Optimized Query: """

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

    def optimize_query(self, state: State):
        prompt = self.prompt_template.invoke({"query": state["question"]})

        if self.model_name == "cohere":
            response = self.llm.invoke(prompt)
            return {"optimized_query": response.content}
        else:
            response = self.llm.generate_content(prompt.to_string())
            return {"optimized_query": response.text}
