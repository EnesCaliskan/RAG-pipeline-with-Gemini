import os
import google.generativeai as genai
import pickle
from pathlib import Path
import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .parser import parse_documents, Document

INDEX_PATH = Path("faiss_index.bin")
CHUNK_MAP_PATH = Path("chunk_map.pkl")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = Path("data")

class VectorRetriever:
    def __init__(self):
        logger.info("Initializing VectorRetriever...")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not found in .env file")

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        self.index = None
        self.chunk_map = {} # Maps index ID to text chunk
        logger.info("VectorRetriever initialized.")

        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("VectorRetreiver initialized with Google Gemini.")


    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into smaller chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.content)
            for chunk in chunks:
                all_chunks.append(Document(content=chunk, metadata=doc.metadata))
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
        return all_chunks
    

    def build_index(self):
        """Builds the FAISS index from documents in the data directory."""
        logger.info("Building FAISS index...")
        if not DATA_DIR.exists():
            logger.error(f"Data directory {DATA_DIR} not found.")
            return
        
        documents = parse_documents(DATA_DIR)
        if not documents:
            logger.error(f"No documents found to index.")
            return
        
        chunks = self._chunk_documents(documents)
        logger.info("Generating embeddings for all chunks...")
        embeddings = self.model.encode([chunk.content for chunk in chunks], show_progress_bar=True)
        
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings).astype('float32'))

        # Create a mapping from index position to the original chunk
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

        # Save the index and the map to disk
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(CHUNK_MAP_PATH, "wb") as f:
            pickle.dump(self.chunk_map, f)
        logger.info(f"Index saved to {INDEX_PATH} and chunk map to {CHUNK_MAP_PATH}.")


    def load_index(self):
        """Loads the FAISS index and chunk map from disk."""
        if INDEX_PATH.exists() and CHUNK_MAP_PATH.exists():
            logger.info("Loading index from the disk...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(CHUNK_MAP_PATH, "rb") as f:
                self.chunk_map = pickle.load(f)
            logger.info(f"Index and chunk map loaded successfully. Index contains {self.index.ntotal} vectors.")
        else:
            logger.warning("Index files not found. Please build the index first.")


    def search(self, query: str, k: int=5) -> list[Document]:
        """
        Searches the index for the most relevant chunks to a query.
        """
        if self.index is None:
            logger.error("Index is not built or loaded. Cannot perform search")
            return []
        
        logger.info(f"Performing search for query: '{query}'...")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)

        results = []
        for i in indices[0]:
            if i != -1: # FAISS returns -1 for empty slots
                results.append(self.chunk_map[i])

        logger.info(f"Found {len(results)} relevant results")
        return results


    def generate_response(self, query: str, chunks: list[Document]) -> str:
        # Generate response using LLm
        # Combine the content of all the chunks into a single string for the context

        context = "\n---\n".join([chunk.content for chunk in chunks])

        # Tell LLM how to behave
        prompt_template = f"""
        You are a helpful assistant who answers questions based ONLY on the provided context.
        Your goal is to provide a clear and concise answer.
        If the information to answer the question is not in the context, you MUST say "I cannot answer this question based on the provided information."

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """

        logger.info("Generating response with LLM...")
        
        try:
            response = self.llm.generate_content(prompt_template)
            return response.text
        except Exception as e:
            logger.error(f"Error during the LLM generation: {e}.")
            return "An error has occured while generating response"

    
        
        


        


    
