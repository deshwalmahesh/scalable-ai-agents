Objective:  
Build an AI-powered assistant that can answer questions based only on uploaded documents.

 

Requirements:

 

1. Create a REST API using Python (FastAPI is preferred).

2. Allow users to upload one or more documents in the following formats:  
      - PDF  
      - DOCX  
      - TXT

3. Extract the document content and prepare it for AI processing.

4. Store the processed data using a vector database of your choice (FAISS, ChromaDB, Pinecone, Qdrant, etc.).

5. Implement a Retrieval-Augmented Generation (RAG) pipeline.

6. Create an endpoint that receives a user's question and returns:  
      - The AI-generated answer.  
      - The document(s) used to generate the answer.  
      - The relevant text snippet or reference.

7. If the answer is not available in the uploaded documents, the assistant must clearly indicate that no relevant information was found instead of generating an incorrect answer.

8. Write clean, maintainable, and well-structured code following good software engineering practices.

9. Include a README that explains:  
      - Project architecture.  
      - Technologies used.  
      - Setup instructions.  
      - How to run the project.  
      - Any assumptions or limitations.

 

Deliverables:

- Complete source code.
- README.
- requirements.txt.
- API documentation (Swagger or Postman collection).
- Sample documents for testing.

 

Engineering:

- Docker support.
- Authentication.
- Conversation history.
- Streaming responses.
- Tests: Integration, Unit, Smoke
- Simple web interface.

 

Evaluation Criteria:

- Code quality.
- Architecture and project structure.
- AI pipeline implementation.
- RAG implementation.
- API design.
- Error handling + Logging
- Documentation.
- Overall solution quality. 
- Scalability
- CI/CD
- Concurrancy
- Config, Centralization, Modularization, Extensibility, Maintainability
- Logic Seperation, Best Practices
