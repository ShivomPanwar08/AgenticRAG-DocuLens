from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# ---------------------------------------------------------
# 1. Document Relevance Grader (Lenient for lists/resumes/policies)
# ---------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance assessment score: 'yes' if relevant, 'no' if irrelevant."
    )
    reason: str = Field(description="Explanation of grading decision.")

def get_document_grader():
    structured_llm = llm.with_structured_output(GradeDocuments)

    system_prompt = (
        "You are an expert document relevance grader.\n"
        "Assess whether the retrieved document contains ANY information, keywords, topics, "
        "or facts that relate to or help answer the user question.\n\n"
        "Scoring Rules:\n"
        "- If the user asks about skills, qualifications, degrees, experience, tools, or projects, "
        "and the chunk contains ANY related technologies (e.g. Python, SQL, Power BI, Excel), skills, or metrics, score 'yes'.\n"
        "- If the user asks about company policy, attendance, late arrival, leaves, or grace periods, "
        "and the chunk discusses timings, penalties, deductions, or approvals, score 'yes'.\n"
        "- Do NOT require full narrative paragraphs; bullet lists and technical profiles count as valid content.\n"
        "- Only assign 'no' if the chunk is completely unrelated (e.g., corporate contact info, office address)."
    )

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {question}\n\nRetrieved context chunk:\n{document}"),
    ])
    return grade_prompt | structured_llm

# ---------------------------------------------------------
# 2. History-Aware Query Rewriter
# ---------------------------------------------------------

class RewrittenQuery(BaseModel):
    improved_query: str = Field(
        description="An optimized, standalone search query containing specific semantic keywords."
    )

def get_query_rewriter():
    structured_llm = llm.with_structured_output(RewrittenQuery)

    system_prompt = (
        "You are an expert AI search query optimizer.\n"
        "Your task is to transform a user question into a standalone, keyword-rich search query.\n"
        "- Use the recent conversation history to resolve pronouns ('he', 'she', 'they', 'it', 'this candidate').\n"
        "- Never return generic evaluation words like 'acceptance confirmation'. Use concrete domain keywords "
        "(e.g., if asking 'should I approve them?', rewrite to 'educational background degrees work experience technical skills').\n"
        "- Strip out meta-instructions like 'From Download.pdf fetch...' and focus strictly on the subject."
    )

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Conversation History:\n{chat_history}\n\nUser Question: {question}\n\nOptimized search query:"),
    ])
    return rewrite_prompt | structured_llm

# ---------------------------------------------------------
# 3. Hallucination / Factual Grounding Grader
# ---------------------------------------------------------

class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if answer is supported by context, 'no' if fabricated."
    )
    explanation: str = Field(description="Explanation of factual grounding.")

def get_hallucination_grader():
    structured_llm = llm.with_structured_output(GradeHallucinations)

    system_prompt = (
        "You are an evaluator assessing whether an answer is grounded in the provided context documents.\n"
        "Score 'yes' if the facts mentioned in the answer are supported by or derived from the context.\n"
        "Score 'no' only if the answer makes up unmentioned facts, dates, or rules."
    )

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context documents:\n\n{documents}\n\nGenerated answer:\n\n{generation}"),
    ])
    return hallucination_prompt | structured_llm