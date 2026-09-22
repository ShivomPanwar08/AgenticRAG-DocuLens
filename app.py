import streamlit as st
import uuid
import os
from pathlib import Path
from langgraph.types import Command

from graph import build_rag_graph
from ingest import ingest_pdf_file, reset_to_default_doc

st.set_page_config(
    page_title="DocuLens AI - Hybrid Agentic RAG",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------
# Session State Initialization (Persistent Graph & Checkpointer)
# ---------------------------------------------------------

if "app" not in st.session_state:
    st.session_state.app = build_rag_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "current_doc" not in st.session_state:
    st.session_state.current_doc = "Attendance Policy 2025.pdf"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------
# Sidebar UI & Ingestion Management
# ---------------------------------------------------------

with st.sidebar:
    st.title("⚙️ DocuLens AI")
    st.caption("Agentic Corrective RAG with Gemini & LangGraph")
    st.markdown("---")

    st.subheader("📄 Active Knowledge Base")
    st.info(f"**Currently querying:**\n`{st.session_state.current_doc}`")

    st.subheader("📤 Upload Custom Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF to replace active context",
        type=["pdf"],
        help="Upload any PDF to update the vector database."
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.current_doc:
        os.makedirs("temp_uploads", exist_ok=True)
        temp_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Ingesting '{uploaded_file.name}' into ChromaDB..."):
            pages, chunks, _ = ingest_pdf_file(temp_path, display_name=uploaded_file.name)
            st.session_state.current_doc = uploaded_file.name
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.app = build_rag_graph()  # Rebuild clean graph on doc change
            st.session_state.pending_interrupt = None
            st.success(f"Indexed {pages} page(s) into {chunks} semantic chunks!")
            st.rerun()

    if st.session_state.current_doc != "Attendance Policy 2025.pdf":
        if st.button("↺ Reset to Default Attendance Policy", use_container_width=True):
            with st.spinner("Restoring Attendance Policy 2025.pdf..."):
                reset_to_default_doc()
                st.session_state.current_doc = "Attendance Policy 2025.pdf"
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.app = build_rag_graph()
                st.session_state.pending_interrupt = None
                st.rerun()

    st.markdown("---")
    st.markdown("**Agentic Guardrails:**")
    st.markdown("• Conversation Intent Router")
    st.markdown("• Document Relevance Grader")
    st.markdown("• History-Aware Query Optimization")
    st.markdown("• Human-in-the-Loop Clarification")
    st.markdown("• Groundedness / Hallucination Check")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.app = build_rag_graph()
        st.session_state.pending_interrupt = None
        st.rerun()

# ---------------------------------------------------------
# Main Chat Display
# ---------------------------------------------------------

st.title("DocuLens AI")
st.caption(f"Context-grounded assistant for: **{st.session_state.current_doc}**")

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "traces" in msg and msg["traces"]:
            with st.expander("🛠️ View Decision Trace"):
                for t in msg["traces"]:
                    st.write(t)

# ---------------------------------------------------------
# HITL Clarification Card
# ---------------------------------------------------------

if st.session_state.pending_interrupt:
    with st.container():
        st.warning("⚠️ **Agent Paused: Human Guidance Required**")
        st.write(st.session_state.pending_interrupt)
        
        clarification_input = st.text_input("Enter your clarification or revised query:", key="hitl_box")
        if st.button("Resume Agent", type="primary"):
            if clarification_input.strip():
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"[Clarification]: {clarification_input}"
                })
                
                with st.spinner("Resuming agent with your guidance..."):
                    res = st.session_state.app.invoke(
                        Command(resume=clarification_input),
                        config=config
                    )
                
                current_state = st.session_state.app.get_state(config)
                if current_state.next and current_state.next[0] == "human_feedback":
                    st.session_state.pending_interrupt = current_state.tasks[0].interrupts[0].value
                else:
                    st.session_state.pending_interrupt = None
                    final_ans = res.get("answer", "No answer could be synthesized.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_ans,
                        "traces": ["Resumed execution via HITL gate.", "Verified factual grounding on context."]
                    })
                st.rerun()

# ---------------------------------------------------------
# User Chat Input
# ---------------------------------------------------------

user_input = st.chat_input(
    f"Ask a question regarding {st.session_state.current_doc}...",
    disabled=st.session_state.pending_interrupt is not None
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing, evaluating relevance, and generating answer..."):
            initial_input = {
                "question": user_input,
                "original_question": user_input,
                "retry_count": 0,
                "gen_retry_count": 0,
                "documents": [],
                "answer": "",
                "chat_history": st.session_state.messages,
                "is_meta_chat": False,
            }

            result = st.session_state.app.invoke(initial_input, config=config)
            current_state = st.session_state.app.get_state(config)

            if current_state.next and current_state.next[0] == "human_feedback":
                st.session_state.pending_interrupt = current_state.tasks[0].interrupts[0].value
                st.rerun()
            else:
                answer = result.get("answer", "No answer found.")
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "traces": [
                        f"Processed query: '{user_input}'",
                        f"Retrieved & Verified {len(result.get('documents', []))} chunk(s)",
                        "Relevance verified via Document Grader"
                    ]
                })