import os
from pathlib import Path
from typing import List
from pypdf import PdfReader
from loguru import logger


class Document:
    def __init__(self, content:str, metadata: str):
        self.content = content
        self.metadata = metadata


    def __repr__(self):
        return f"Document(metadata={self.metadata})"
    

def parse_documents(data_path: Path) -> list[Document]:
    """
    Parses all .txt and .pdf files in the specified directory.

    Args:
    data_path: The path to the directory containing the documents.

    Returns:
    A list of Document objects.
    """
        
    if not data_path.is_dir():
        logger.error(f"Provided path '{data_path}' cannot be found in the directory.")
        return []
    
    documents = []
    logger.info(f"Parsing documents from '{data_path}'...")


    for file_path in data_path.iterdir():
        try:
            if file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                metadata = {"source": file_path.name}
                documents.append(Document(content, metadata))
                logger.success(f"Succesfully parsed {file_path.name}")
            
            elif file_path.suffix == ".pdf":
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() or ""
                
                metadata = {"source": file_path.name}
                documents.append(Document(content, metadata))
                logger.success(f"Successfully parsed {file_path.name}")


        except Exception as e:
            logger.error(f"Failed to parse")

        
        logger.info(f"Finished parsing. Total documents: {len(documents)}")
    return documents