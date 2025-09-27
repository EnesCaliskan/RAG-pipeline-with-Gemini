from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from loguru import logger

from .core.retriever import VectorRetriever, INDEX_PATH
from .schemas import QueryRequest, QueryResponse, DocumentResponse, AnswerResponse


# Initialize the FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="An API querying documents using RAG pipeline",
    version="1.0.0"
)

# Initialize the retriever
retriever = VectorRetriever()

@app.on_event("startup")
async def startup_event():
    """
    On startup, check if the index exists. If not, build it.
    If it exists, load it.
    """
    if not INDEX_PATH.exists():
        logger.warning("Index not found. Building index from scratch. This may take a while...")
        retriever.build_index()
    else:
        retriever.load_index()


@app.post("/query", response_model=AnswerResponse)
def query_documents(request: QueryRequest):
    """
    Accepts a user query and returns the most relevant document chunks.
    """
    if retriever.index is None:
        raise HTTPException(status_code=503, detail="Index not ready. Please try again.")
    
    try:
        # Without MultiQuery
        # relevant_chunks = retriever.search(query=request.query, k=5)
        relevant_chunks = retriever.search_with_multi_query(query=request.query)

        if not relevant_chunks:
            return AnswerResponse(
                query=request.query,
                answer="No relevant information has been found in the documents for this question..."
            )
        
        # Checking the relevant chunks based on the query
        # for i, doc in enumerate(relevant_chunks, start=1):
        #     logger.info(f"Chunk {i}: {doc.content[:200]}...")

        # Kullanicinin querysi ile relevant chunklari LLM'e yolladik
        answer = retriever.generate_response(query=request.query, chunks=relevant_chunks)
        return AnswerResponse(query=request.query, answer=answer)
    
    except Exception as e:
        logger.error(f"An error has occured during the query processing {e}.")
        raise HTTPException(status_code=500, detail= "An error has occured during the query processing.")


        #RAG pipeline without LLM
        """
        results = retriever.search(query=request.query, k=5)
        # Convert Document objects to DocumentResponse Pydantic models
        response_chunks = [DocumentResponse(content=doc.content, metadata=doc.metadata) for doc in results]

        return QueryResponse(query=request.query, relevant_chunks=response_chunks)
    except Exception as e:
        logger.error(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
        """

@app.get("/")
def read_root():
    return {"message": "Welcome."}

