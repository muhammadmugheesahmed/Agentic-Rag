import os
from pathlib import Path
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")

# ==========================================
# 1. Load PDFs from docs/ & Build Vector Store
# ==========================================

DOCS_DIR = Path("docs")
if not DOCS_DIR.exists() or not list(DOCS_DIR.glob("*.pdf")):
    raise FileNotFoundError(
        "No PDF files found in 'docs/'. Please create a 'docs/' folder and add at least one PDF."
    )

print("Loading PDFs from docs/ ...")
loader = PyPDFDirectoryLoader(str(DOCS_DIR))
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_splits = splitter.split_documents(raw_docs)

print(f"Indexed {len(doc_splits)} chunks from {len(raw_docs)} PDF page(s).")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(doc_splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_pdf_docs",
    description=(
        "Search and retrieve relevant sections from the uploaded PDF documents in docs/. "
        "Use this whenever the user asks a question about the document content."
    ),
)
tools = [retriever_tool]


# ==========================================
# 2. Define Agentic RAG State & Nodes
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    rewrite_count: int
    steps: list[str]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def extract_text(content) -> str:
    """Safely extract plain text from Gemini's string or list-of-blocks response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        ).strip()
    return str(content)

def agent_node(state: AgentState):
    """Decides whether to call the PDF retriever tool or answer directly."""
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "steps": state.get("steps", []) + ["Agent analyzed query"],
    }


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Grades whether the retrieved PDF chunks are relevant to the user's question."""
    if state.get("rewrite_count", 0) >= 2:
        return "generate"

    grader_llm = llm.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing the relevance of retrieved PDF excerpts to a user question.\n\n"
        "Retrieved excerpts:\n{context}\n\n"
        "User question: {question}\n\n"
        "If the excerpts contain information related to the question, grade them as relevant. "
        "Return 'yes' or 'no'."
    )

    question = state["question"]
    last_message = state["messages"][-1]  # ToolMessage from retriever

    result = (prompt | grader_llm).invoke(
        {"question": question, "context": last_message.content}
    )

    if result.binary_score.strip().lower() == "yes":
        return "generate"
    return "rewrite"


def rewrite_node(state: AgentState):
    """Rewrites the query to improve PDF retrieval."""
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "Reformulate the following user question into a clearer, keyword-rich search query "
        "to retrieve better results from a PDF document.\n\n"
        "Original question: {question}\n"
        "Improved query:"
    )
    improved_query = (prompt | llm).invoke({"question": question})
    rewritten_text = extract_text(improved_query.content)
    return {
        "messages": [HumanMessage(content=rewritten_text)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "steps": state.get("steps", []) + [f"Rewrote query: {rewritten_text}"],
    }


def generate_node(state: AgentState):
    """Generates the final answer grounded in the retrieved PDF context."""
    question = state["question"]
    context = state["messages"][-1].content

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful PDF assistant. Answer the user's question clearly and accurately "
        "using ONLY the provided PDF context. If the answer is not in the context, say so.\n\n"
        "PDF Context:\n{context}\n\n"
        "Question: {question}"
    )
    response = (prompt | llm).invoke({"context": context, "question": question})
    return {
        "messages": [response],
        "steps": state.get("steps", []) + ["Generated grounded answer from PDF"],
    }


# ==========================================
# 3. Build & Compile LangGraph Workflow
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", ToolNode(tools))
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)
workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)

rag_app = workflow.compile()


# ==========================================
# 4. FastAPI Server
# ==========================================

app = FastAPI(title="Gemini Agentic RAG Chatbot")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    steps: list[str]


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = rag_app.invoke(
            {
                "messages": [HumanMessage(content=user_msg)],
                "question": user_msg,
                "rewrite_count": 0,
                "steps": [],
            }
        )
        final_answer = extract_text(result["messages"][-1].content)
        steps = result.get("steps", [])
        return ChatResponse(answer=final_answer, steps=steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("RAG:app", host="127.0.0.1", port=8000, reload=True)