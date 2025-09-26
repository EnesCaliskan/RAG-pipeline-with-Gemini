from langchain_core.tools import Tool
from .retriever import VectorRetriever

retriever = VectorRetriever()
# Load or build index
retriever.load_index() or retriever.build_index()

# Wrapping retriever's search in LangChain
document_search_tool = Tool(
    name="document_searcher",
    func=retriever.search,
    description="Use this tool to find information and answer questions about the content within the provided PDF and TXT documents. Provide a detailed query as input."
)

# Creating list of all tools our agent will have access to
agent_tools = [document_search_tool]