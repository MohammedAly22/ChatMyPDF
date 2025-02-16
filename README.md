# ChatMyPDF - Chat with Your PDFs using RAG

## 🚀 Overview

ChatMyPDF is a **Retrieval-Augmented Generation (RAG) application** that allows users to upload PDF documents and interact with them using an AI-powered chatbot. It uses **embedding-based retrieval** to extract relevant information from PDFs and **LLM-powered generation** to provide context-aware responses.

## 📜 Table of Contents

- [Architecture](#-Architecture)
- [Features](#-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)

---

## 🏗️ Architecture

The application follows a **RAG (Retrieval-Augmented Generation) pipeline**, consisting of the following steps:

1. **Document Upload**: Users upload a PDF file.
2. **Text Extraction & Chunking**: The text is extracted and split into meaningful chunks.
3. **Embeddings Generation**: Each chunk is converted into vector embeddings using `sentence-transformers/all-mpnet-base-v2`.
4. **Indexing**: The embeddings are stored in a vector database for fast retrieval.
5. **Query Optimization**: User queries are optimized for better retrieval.
6. **Retrieval & Contextual Response**: The system retrieves the most relevant text chunks and generates a response using an LLM (Cohere/Gemini).

### Basic RAG Architecture
Here is the most basic RAG architecture:
![basic_rag_animation](https://github.com/user-attachments/assets/36f89f7e-ee63-4db5-8e5f-32b2fcd82467)

### ChatMyPDF - Version 1.0.0
This version contains the basic RAG implementation leveraging:
- `InMemoryVectorStore` for easily store the embeddings.
- `RecursiveCharacterTextSplitter` for chunking.
![version_1 0 0_animation](https://github.com/user-attachments/assets/add556e1-a756-4df5-8d35-a53750e15ed5)

### ChatMyPDF - Version 1.1.0
This version contains a little bit of advanced RAG implementation compared to `v1.0.0` including:
- Using `ChromaDB` vector database instead of `InMemoryVectorStore`.
- Using `PyMuPDFLoader` for data loading instead of `PyPDFLoader`.
- Using `SemanticChuncker` for a more advanced chunking strategy. 
- Using `similarity_search_with_relevance_scores` function for retrieval.
- Adding the functionality of showing the retrieved context.
![version_1 1 0_animation](https://github.com/user-attachments/assets/12cf55f5-61e7-4dd9-853b-7f38fb5c978a)

### ChatMyPDF - Version 1.2.0
This is the final version (up until now) that contains extra advanced modules for enhancing the performance:
- Implemented `query_optimization` module responsible for enhancing the query leveraging LLMs.
- Leveraging `re-ranking` module responsible for re-rank the documents after `retrieval` to better filter
out irrelevant context and ensure that only the relevant contexts is passed to the `generator` module.
![version_1 2 0_animation](https://github.com/user-attachments/assets/a8ad3e88-5cec-4813-82f2-b1dbe89fbe5b)

---

## ✨ Features

✅ Upload PDF documents and interact with them using natural language queries.\
✅ Supports multiple AI models (`Cohere`, `Gemini`).\
✅ Uses `sentence-transformers/all-mpnet-base-v2` for semantic search.\
✅ Efficient semantic text chunking and vector-based retrieval for accurate responses.\
✅ Optimize the user queries for better retrieval results.\
✅ Leveraging document re-ranking technique for better savings in the context window.\
✅ Interactive Streamlit UI for easy use.

---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/MohammedAly22/ChatMyPDF.git
cd ChatMyPDF
```

### **2️⃣ Create and Activate Virtual Environment**

```sh
python -m venv rag-app-env
source rag-app-env/bin/activate  # On macOS/Linux
rag-app-env\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**

Open the `api_keys.json` file in the root directory and add your API keys:

```JSON
{
  "LangSmith": "YOUR_LANGSMITH_API_KEY"
  "Cohere": "YOUR_COHERE_API_KEY"
  "Gemini": "YOUR_GOOGLE_API_KEY"
}

```

### **5️⃣ Run the Application**

```sh
streamlit run src/app.py
```

### **6️⃣ View the Interface**
After following the above instructions, you may expect to see this interface:

![image](https://github.com/user-attachments/assets/402d17b5-e597-474f-8085-739c1b3a14cb)

---

## 📖 Usage

1. Open the app in your browser (default: `http://localhost:8501`).
2. Upload a PDF file.
3. Choose a model (`Cohere` or `Gemini`).
4. Decide to show the contexts and/or re-ranking results.
5. Ask questions about the document.
6. View AI-generated responses with retrieved context.

Here is a demonstrated screenshot asking some questions regarding my resume:

![image](https://github.com/user-attachments/assets/096608da-4ae2-483f-be1f-8853d1cca34c)
  

---

## 💡 Technologies Used

- **Streamlit** - Interactive UI for the application.
- **LangChain** - RAG pipeline, query optimization, and document re-ranking.
- **LangGraph** - For building state graph RAG pipeline.
- **LangSmith** - For monitoring and inspecting the logs of the intermediate steps.
- **Sentence-Transformers** - Embeddings for retrieval.
- **Cohere / Gemini** - LLM for response generation.
- **InMemoryVectorStore / ChromaDB** - Vector database for document indexing.

---

## 🔮 Future Enhancements

- ✅ Integration with OpenAI GPT models.
- ✅ Multi-PDF query support.
- ✅ Multi-Modal RAG questions including images and tables.
- ✅ Fine-tuned retrieval using metadata filters.

---

### **💬 Have Questions?**

Reach out on GitHub or open an issue!

---

🎯 **ChatMyPDF - Your AI-powered PDF Assistant!** 🚀
