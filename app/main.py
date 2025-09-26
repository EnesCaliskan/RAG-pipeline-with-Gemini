from fastapi import FastAPI, HTTPException
from loguru import logger

from .core.retriever import VectorRetriever, INDEX_PATH
from .schemas import QueryRequest, QueryResponse, DocumentResponse


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


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Accepts a user query and returns the most relevant document chunks.
    """
    if retriever.index is None:
        raise HTTPException(status_code=503, detail="Index not ready. Please try again.")
    
    try:
        results = retriever.search(query=request.query, k=5)
        # Convert Document objects to DocumentResponse Pydantic models
        response_chunks = [DocumentResponse(content=doc.content, metadata=doc.metadata) for doc in results]

        return QueryResponse(query=request.query, relevant_chunks=response_chunks)
    except Exception as e:
        logger.error(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome."}

