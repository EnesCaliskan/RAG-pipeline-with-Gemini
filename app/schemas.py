from pydantic import BaseModel, Field
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="This user's query text.")


class DocumentResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    relevant_chunks: List[DocumentResponse]

class AnswerResponse(BaseModel):
    query: str
    answer: str

