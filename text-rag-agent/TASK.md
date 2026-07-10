# TASK:
Using the BEST (the very vest in industry), the cleanest ones to show the applicability of the Engineering best practices for scalability, security, maintainability, concurrancy etc etc etc, build this app below

For Reference: you can use the exisitng tools as much as possible like fastapi-users, tenacity (I dont know what that is but people say it's useful), some library for rate limiting etc etc. Same with Langchain/OpenaI API to set the models on demand. for infra docker based, minimal one like Redis, Some vector DB etc. 


You can look at ~/Desktop/Personal/multi agent orchestrator.  It has LOTS and LOTS of stuff bit instead of building all of stuff ourself, we cna take "INSPIRATION" or copy paste exact modules which we need. Onmly and only needed things.

It means Docker, kubernetes, Caching, user management, etc etc everythign a production grade system has. This app is supposed to have the needed features (whihc might be basic) but shows the strength of python ecosystem librarties and most importantly the design pattern and engineerign best practices.

It has many and many and many things. directly use the stuff we want and need. This project has some extra things too which we need to build

# Note:
Things must be centralised. All the configs and common tolls in common things.
always see if therei s somethign we can use immediately. If not then is there we can reuse with splitting or args etc. and even if not possible then look for the new code as the last resory that too modulat that can be extended in future. 

Use classes and their function. 

**MUST DO**: Never ever ever never make ANY function of class or file etc just for the sake of it. Plan first on why you are doing it, is there something we an reuse or extend etc and if you have to build it, then what will be the effect. make it generalizable. 

No overengineering. nothing fancy. just robust salable code. Precuisely. Step by step. 

 
# UI
All the Needed components only. no bullshit. no fluff. No overwngineering. beautiful looking UI and code

Objective:  
Build an AI-powered assistant that can answer questions based only on uploaded documents.

 

Scenario:  
A company wants an AI assistant that allows users to upload documents and ask questions about their contents. The assistant should answer only from the uploaded documents and should not generate information that does not exist in them.

 

# Requirements:

 

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

 

## Deliverables:

- Complete source code.
- README.
- requirements.txt.
- API documentation (Swagger or Postman collection).
- Sample documents for testing.

 

## Engineering:

- Docker support.
- Authentication.
- Conversation history.
- Streaming responses.
- Tests: Integration, Unit, Smoke
- Simple web interface.

 

## Evaluation Criteria:

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
