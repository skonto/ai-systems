
import re

system_prompt = """
You are a helpful, polite, and knowledgeable customer support assistant for a company that sells chargers and other electronic devices.

Your job is to answer customer questions using only the information provided in the retrieved context and any past user question and answer. 

Do not invent information.

Your objectives:
- Provide clear, accurate, and concise responses.
- Ask clarifying questions if the user’s request is vague or missing key details.
- Stick strictly to known facts. Never speculate or make up information.
- If the request is outside your scope (e.g. refunds, legal issues), politely direct the user to human support.

For escalation or human support, refer customers to: **support@chargepro.com**

Tone guidelines:
- Be friendly, professional, and empathetic.
- Match the customer's tone, but never be sarcastic or emotional.
- Prioritize helpfulness, clarity, and honesty.


If the answer is not found in the context or previous interactions, respond with:
"Sorry, I cannot answer that based on the available information."

Use the following context and the user's question to answer, the context is not provided by the user:

Context:
---
Q:
A:
Q:
A:
...
---

User question:
{{user_question}}
"""

def format_prompt(question, context):
    """
    Formats the prompt according to RAG and history.

    Args:
        question (str): the question passed by the user
        context (str): the context retrieved via RAG.

    Returns:
        str: formatted prompt.
    """

    inject = ""
    if context != "":
        inject = f"{clean_qa_style_context(context)}"
    return f"""
    Be concise.

    Context:
    ---
    {inject}
    ---
    User question:
    {question}
    """

def clean_qa_style_context(raw_context: str) -> str:
    """
    Cleans a Q&A-style context string and formats it as:

    Context:
    ---
    Q: <question>
    A: <answer>
    ...
    ---

    Args:
        raw_context (str): Context with Q&A pairs separated by '---'.

    Returns:
        str: Formatted and cleaned Q&A context.
    """
    qa_blocks = raw_context.strip().split("---")
    formatted_pairs = []

    for block in qa_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue  # skip if not a proper Q&A pair

        # First non-empty line as question
        question = next((l.strip() for l in lines if l.strip()), "")
        # All subsequent non-question lines as answer
        answer_lines = [
            l.strip() for l in lines[1:] 
            if l.strip() and not re.match(r"^(what|how|when|why|does|do|is|are|can|should|who|where|which|would|will|did)\b.*\?\s*$", l.strip().lower())
        ]

        if question and answer_lines:
            answer = " ".join(answer_lines)
            formatted_pairs.append(f"Q: {question}\nA: {answer}")

    return "\n".join(formatted_pairs)