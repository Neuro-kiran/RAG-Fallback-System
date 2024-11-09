# Enhanced Retrieval-Augmented Generation (RAG) Assistant with Fallback Mechanism

## Demo Video

Check out the demo of the project in action:

[![Watch the Demo Video](https://img.youtube.com/vi/70924e5d16cd4c47a4240e8ab8a8dd6e/maxresdefault.jpg)](https://www.loom.com/share/70924e5d16cd4c47a4240e8ab8a8dd6e?sid=bcff8f64-433e-4525-bf27-d1fd0397e53a)

Alternatively, you can download the demo video from the repository:
[Download Demo Video (MP4)](Demo.mp4)

---
This project implements an Enhanced Retrieval-Augmented Generation (RAG) assistant with an intelligent fallback mechanism. The assistant utilizes LangChain, HuggingFace, and OpenAI's ChatGroq model to respond to user queries. If the LLM (Large Language Model) lacks sufficient confidence in answering, it activates a retrieval system to fetch additional data from documents. 

The core idea is that the LLM first evaluates its confidence and only invokes the retrieval system if needed. This ensures efficient use of resources while maintaining response accuracy.

## Features

- **Document Upload**: Upload PDF or text files to process and use as the knowledge base.
- **Confidence-based Fallback**: The LLM self-assesses its knowledge and invokes the retrieval system only when necessary.
- **RAG System**: Combines document retrieval with language generation to provide relevant responses.
- **Contextual Querying**: Users can ask questions, and the assistant will use document context to provide accurate answers.
- **Easy Deployment**: Built with Streamlit for a simple, interactive web interface.

## Requirements

Ensure you have the following libraries installed. You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
