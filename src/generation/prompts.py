"""
Prompt Engineering for RAG

Carefully crafted prompts to:
1. Ground answers in retrieved context
2. Require citations
3. Refuse when info is missing
4. Output structured JSON
"""

"""
    Build the full RAG prompt.
    
    Structure:
    1. System instructions (grounding rules)
    2. Context chunks with citations
    3. User query
    4. Output format requirements
    
    Args:
        query: User's question
        retrieved_chunks: List of chunk texts
        metadata_list: List of metadata dicts for each chunk
        
    Returns:
        Complete prompt string
"""
def build_rag_prompt(query:str, retrieved_chunks: list,
                     metadata_list: list) -> str:
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata_list), 1):
        source_info = f"Source {i}: {meta['company']} 10-K({meta['filing_date']})"
        context_parts.append(f"[{i}] {source_info}\n{chunk}\n")

    context = "\n".join(context_parts)

    prompt = f"""You are a financial analyst assistant.
    Your task is to answer questions STRICTLY based on the provided 10-K filing excerpts.

CRITICAL RULES:
1. ONLY use information from the context below
2. ALWAYS cite sources using [1], [2], etc.
3. If the context doesn't contain enough information, say 
"I don't have enough information in the provided documents to answer this question"
4. Be precise and factual
5. Quote exact numbers when available
6. Quote exact figures, percentages, and financial numbers when available.
7. Do NOT infer, assume, or use outside knowledge.
8. The "citations" array MUST contain every citation number referenced in the answer.

CONTEXT (SEC 10-K EXCERPTS):
{context}

USER QUESTION:
{query}

Provide your answer in the following JSON format:
{{
  "answer": "Your detailed answer with citations [1], [2], etc."
  "citations": [1, 2], //List of source numbers used
  "confidence": "high|medium|low",
  "has_sufficient_info": true/false
}}

ANSWER:"""
    
    return prompt

SYSTEM_PROMPT = """You are a precise financial analyst assistant.
You answer questions ONLY using the provided context.
You ALWAYS cite your sources.
You NEVER use external knowledge.
You admit when you don't have enough information."""

def build_chat_messages(query: str, retrieved_chunks: list,
                        metadata_list: list) -> list:
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata_list), 1):
        source_info = f"{meta['company']} 10-K ({meta['filing_date']})"
        context_parts.append(f"[{i}] {source_info}\n{chunk}")

    context = "\n\n".join(context_parts)
    user_message = f"""Context from SEC 10-K filings:

{context}

Question: {query}

Please answer based ONLY on the context above.
Do NOT use external knowledge.
Do NOT infer beyond what is explicitly stated.
Cite sources using [1], [2], etc.
If you don't have enough information, say so clearly.
If insufficient information is available, respond exactly:
   "I don't have enough information in the provided documents to answer this question."
The citations array MUST include every citation referenced in the answer.

Respond in JSON:
{{
  "answer": "...",
  "citations": [1, 2],
  "confidence": "high/medium/low",
  "has_sufficient_info": true/false
}}"""
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role":"user", "content": user_message}
    ]