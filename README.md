## About the App

RAG Chatbot is a fast, document-aware AI assistant that lets users upload a PDF and ask natural-language questions about its contents. The app uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve the most relevant information from the uploaded document and generate grounded answers instead of relying on generic model memory.

Behind the scenes, the system automatically processes the uploaded PDF, extracts and chunks the text, converts it into embeddings, stores it in a FAISS vector index, and retrieves the best-matching context for every query. That retrieved context is then passed to a Groq-hosted LLM, enabling responses that are both relevant and low-latency.

This project demonstrates how modern GenAI systems can be built in a practical, user-friendly way using Streamlit, LangChain, Hugging Face embeddings, FAISS, and Groq inference. It is designed as both a useful document-chat application and a strong showcase of applied RAG architecture, semantic retrieval, and production-oriented AI app development.




## Workflow Diagram

```mermaid
flowchart TD
    A[User opens Streamlit app] --> B[Upload PDF]
    B --> C[Auto-process uploaded PDF]
    C --> D[Save temporary PDF file]
    D --> E[Extract text with PyPDFLoader]
    E --> F[Split text into chunks]
    F --> G[Generate embeddings with Hugging Face]
    G --> H[Store vectors in FAISS index]
    H --> I[Create retriever and save in session state]
    I --> J[User asks a question]
    J --> K[Retriever finds relevant chunks]
    K --> L[Build grounded prompt with context]
    L --> M[Send prompt to Groq LLM]
    M --> N[Return answer in chat UI]
```

## System Architecture

```mermaid
flowchart LR
    subgraph UI["Frontend Layer"]
        A[Streamlit Chat Interface]
        B[File Uploader]
        C[Chat Input]
    end

    subgraph APP["Application Layer"]
        D[Session State]
        E[Auto PDF Processor]
        F[Prompt Builder]
    end

    subgraph RAG["RAG Pipeline"]
        G[PyPDFLoader]
        H[Text Splitter]
        I[HuggingFace Embeddings]
        J[FAISS Vector Store]
        K[Retriever]
    end

    subgraph LLM["Inference Layer"]
        L[Groq API]
        M[Llama 3.1 8B Instant]
    end

    A --> B
    A --> C
    B --> E
    E --> G
    G --> H
    H --> I
    I --> J
    J --> K
    C --> F
    K --> F
    F --> L
    L --> M
    M --> A
    D --> E
    D --> K
```

## Query Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit App
    participant V as FAISS Retriever
    participant L as Groq LLM

    U->>S: Upload PDF
    S->>S: Extract, chunk, embed
    S->>V: Store vectors
    U->>S: Ask question
    S->>V: Retrieve relevant chunks
    V-->>S: Return top matching context
    S->>L: Send prompt + retrieved context
    L-->>S: Return grounded answer
    S-->>U: Show response in chat
```

## Performance Optimization Flow

```mermaid
flowchart TD
    A[Upload PDF] --> B{Already processed?}
    B -- Yes --> C[Reuse cached retriever]
    B -- No --> D[Process PDF once]
    D --> E[Build FAISS index]
    E --> F[Store in session state]
    C --> G[Ask question]
    F --> G
    G --> H[Retrieve top 2 chunks]
    H --> I[Send compact context to Groq]
    I --> J[Faster response]
```
