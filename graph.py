import os
from typing import List, TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from ingest import build_or_get_vector_store
from evaluators import (
    get_document_grader,
    get_query_rewriter,
    get_hallucination_grader
)

load_dotenv()

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------

class GraphState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    retry_count: int
    gen_retry_count: int
    answer: str
    chat_history: List[dict]
    is_meta_chat: bool


# ---------------------------------------------------------
# 2. Router: History/Meta vs. Document RAG
# ---------------------------------------------------------

def check_query_type(state: GraphState) -> Literal["chat_history_reply", "retrieve"]:
    """Detects if query is purely about past conversation or requires document retrieval."""
    q = (state.get("original_question") or state.get("question") or "").lower()
    
    # Meta / conversation history intent keywords
    meta_phrases = [
        "what did i ask", "what questions did i ask", "previous question", 
        "repeat what i said", "our conversation", "chat history", "who are you"
    ]
    
    if any(phrase in q for phrase in meta_phrases):
        print("--> Router: Detected Chat History / Meta Query. Bypassing vector search.")
        return "chat_history_reply"
    
    return "retrieve"


def chat_history_reply(state: GraphState):
    """Answers directly from conversation history without vector retrieval."""
    print("\n--- [NODE: CHAT HISTORY REPLY] ---")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    history_str = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in state.get("chat_history", [])]
    )
    
    prompt = ChatPromptTemplate.from_template(
        "You are an intelligent assistant. Based strictly on the conversation history below, "
        "answer the user's question clearly.\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    
    chain = prompt | llm
    target_q = state.get("original_question") or state.get("question")
    res = chain.invoke({"chat_history": history_str or "No previous history recorded.", "question": target_q})
    
    return {
        "question": target_q,
        "answer": str(res.content).strip(),
        "documents": []
    }


# ---------------------------------------------------------
# 3. Standard RAG Nodes
# ---------------------------------------------------------

def retrieve(state: GraphState):
    query = state.get("question") or state.get("original_question") or ""
    print(f"\n--- [NODE: RETRIEVE] Query: '{query}' ---")
    
    vector_store = build_or_get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    print(f"    Retrieved {len(docs)} chunk(s).")
    
    return {"documents": docs, "question": query}


def grade_documents(state: GraphState):
    print("\n--- [NODE: GRADE DOCUMENTS] ---")
    grader = get_document_grader()
    target_q = state.get("original_question") or state.get("question", "")
    docs = state.get("documents", [])

    filtered_docs = []
    for idx, doc in enumerate(docs, 1):
        res = grader.invoke({"question": target_q, "document": doc.page_content})
        score = res.binary_score.lower()
        if score == "yes":
            print(f"    Chunk {idx}: RELEVANT (Page {doc.metadata.get('page')})")
            filtered_docs.append(doc)
        else:
            print(f"    Chunk {idx}: IRRELEVANT (Filtered out)")

    return {
        "documents": filtered_docs,
        "question": target_q
    }


def rewrite_query(state: GraphState):
    print("\n--- [NODE: REWRITE QUERY] ---")
    rewriter = get_query_rewriter()
    
    history_str = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in state.get("chat_history", [])[-4:]]
    )
    
    target_q = state.get("original_question") or state.get("question", "")
    res = rewriter.invoke({
        "question": target_q,
        "chat_history": history_str or "No previous conversation."
    })
    
    print(f"    Rewritten to: '{res.improved_query}'")
    return {
        "question": res.improved_query,
        "retry_count": state.get("retry_count", 0) + 1
    }


def human_feedback(state: GraphState):
    print("\n--- [NODE: HUMAN-IN-THE-LOOP INTERRUPT] ---")
    current_q = state.get("question", "your query")
    prompt_message = (
        f"Retrieval could not locate relevant documentation for '{current_q}'. "
        "Please provide additional context or clarify your query:"
    )
    user_clarification = interrupt(value=prompt_message)

    print(f"    [HUMAN RESUMED] Clarification: '{user_clarification}'")
    return {
        "question": str(user_clarification),
        "original_question": str(user_clarification),
        "retry_count": 0
    }


def generate(state: GraphState):
    print("\n--- [NODE: GENERATE ANSWER] ---")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an intelligent document assistant.\n"
            "Answer the question thoroughly using the provided Context and Conversation History.\n"
            "Cite source pages whenever available (e.g., [Page 1])."
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\nContext Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    ])

    formatted_context = "\n\n".join(
        [f"[Page {d.metadata.get('page', 'Unknown')}]: {d.page_content}" for d in state.get("documents", [])]
    )
    
    history_str = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in state.get("chat_history", [])[-4:]]
    )

    target_q = state.get("original_question") or state.get("question", "")
    chain = prompt | llm
    response = chain.invoke({
        "context": formatted_context,
        "chat_history": history_str,
        "question": target_q
    })

    if isinstance(response.content, str):
        clean_text = response.content
    elif isinstance(response.content, list):
        clean_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content])
    else:
        clean_text = str(response.content)

    return {
        "documents": state.get("documents", []),
        "answer": clean_text.strip(),
        "question": target_q,
        "gen_retry_count": state.get("gen_retry_count", 0)
    }


# ---------------------------------------------------------
# 4. Routing & Grounding
# ---------------------------------------------------------

def decide_to_generate(state: GraphState):
    filtered_docs = state.get("documents", [])
    retries = state.get("retry_count", 0)

    if filtered_docs:
        print("--> Decision: Valid context found. Proceeding to GENERATE.")
        return "generate"

    if retries < 1:
        print(f"--> Decision: Context insufficient. Retrying rewrite (Attempt {retries + 1}/1).")
        return "rewrite_query"
    else:
        print("--> Decision: Automated retries exhausted. Routing to HUMAN INTERRUPT.")
        return "human_feedback"


def check_hallucination(state: GraphState):
    print("\n--- [GUARDRAIL: CHECK FACTUAL GROUNDING] ---")
    grader = get_hallucination_grader()
    
    docs_text = "\n\n".join([d.page_content for d in state.get("documents", [])])
    answer = state.get("answer", "")
    
    res = grader.invoke({"documents": docs_text, "generation": answer})
    grounded = res.binary_score.lower() == "yes"
    gen_retries = state.get("gen_retry_count", 0)
    
    if grounded:
        print(f"--> Grounding: PASS. {res.explanation}")
        return "grounded"
    else:
        print(f"--> Grounding: FAIL. {res.explanation}")
        if gen_retries < 1:
            return "not_grounded"
        return "grounded"


# ---------------------------------------------------------
# 5. Graph Assembly
# ---------------------------------------------------------

def build_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("chat_history_reply", chat_history_reply)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("human_feedback", human_feedback)
    workflow.add_node("generate", generate)

    # Initial branch based on intent
    workflow.add_conditional_edges(
        START,
        check_query_type,
        {
            "chat_history_reply": "chat_history_reply",
            "retrieve": "retrieve"
        }
    )

    workflow.add_edge("chat_history_reply", END)
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "human_feedback": "human_feedback",
        }
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("human_feedback", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        check_hallucination,
        {
            "grounded": END,
            "not_grounded": "generate"
        }
    )

    return workflow.compile(checkpointer=MemorySaver())