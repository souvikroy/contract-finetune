"""Custom prompts for legal document analysis."""
from typing import List, Dict, Any


class LegalPrompts:
    """Collection of prompts for legal document analysis."""
    
    SYSTEM_PROMPT = """You are an expert legal document analyst specializing in contract analysis. 
Your role is to help users understand legal contracts, extract specific information, and answer questions 
about contract terms, clauses, and obligations.

Guidelines:
- Provide accurate, clear, and concise answers based on the provided contract context
- Cite specific sections, clauses, or articles when referencing contract terms
- If information is not found in the provided context, clearly state that
- Use legal terminology appropriately but explain complex terms when helpful
- Maintain objectivity and focus on factual information from the contract"""

    CLAUSE_EXTRACTION_PROMPT = """Extract and classify the following legal clauses from the contract context.

Contract Context:
{context}

Task: Identify and extract all relevant clauses related to: {query}

For each clause found, provide:
1. Clause identifier (Article, Section, Clause number)
2. Clause type (commitment, payment, termination, guarantee, etc.)
3. Key obligations or terms
4. Relevant parties involved
5. Page number or location reference

Format your response as a structured list."""

    CLASSIFICATION_PROMPT = """Classify the following contract clause into one of these categories:
- Commitment/Obligation
- Payment/Financial
- Termination/Cancellation
- Guarantee/Security
- Disqualification/Exclusion
- Compensation/Damages
- General/Other

Clause Text:
{clause_text}

Provide:
1. Classification category
2. Confidence level (High/Medium/Low)
3. Reasoning for the classification"""

    QNA_PROMPT = """Answer the following question about the legal contract based on the provided context.

Contract Context:
{context}

Question: {question}

Instructions:
- Base your answer solely on the provided contract context
- Cite specific sections, articles, or clauses when referencing information
- If the answer is not in the context, state "The information is not available in the provided contract context"
- Provide a clear, concise answer suitable for legal professionals

Answer:"""

    SUMMARIZATION_PROMPT = """Summarize the following contract section or clause.

Contract Text:
{text}

Provide a concise summary that includes:
1. Main topic or subject
2. Key obligations or terms
3. Important parties involved
4. Critical dates or deadlines (if any)
5. Notable conditions or exceptions

Summary:"""

    COMPARISON_PROMPT = """Compare the following contract clauses and identify similarities and differences.

Clause 1:
{clause1}

Clause 2:
{clause2}

Provide:
1. Similarities between the clauses
2. Key differences
3. Potential implications of these differences"""

    @staticmethod
    def format_qa_prompt(context: str, question: str) -> str:
        """Format Q&A prompt with context and question."""
        return LegalPrompts.QNA_PROMPT.format(context=context, question=question)
    
    @staticmethod
    def format_clause_extraction_prompt(context: str, query: str) -> str:
        """Format clause extraction prompt."""
        return LegalPrompts.CLAUSE_EXTRACTION_PROMPT.format(context=context, query=query)
    
    @staticmethod
    def format_classification_prompt(clause_text: str) -> str:
        """Format classification prompt."""
        return LegalPrompts.CLASSIFICATION_PROMPT.format(clause_text=clause_text)
    
    @staticmethod
    def format_summarization_prompt(text: str) -> str:
        """Format summarization prompt."""
        return LegalPrompts.SUMMARIZATION_PROMPT.format(text=text)
    
    @staticmethod
    def format_comparison_prompt(clause1: str, clause2: str) -> str:
        """Format comparison prompt."""
        return LegalPrompts.COMPARISON_PROMPT.format(clause1=clause1, clause2=clause2)
