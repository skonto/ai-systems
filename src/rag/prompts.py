import re
from typing import Dict, List

# Pre-compile regex to identify question lines
QUESTION_PATTERN = re.compile(
    r"^(?:what|how|when|why|does|do|is|are|can|should|"
    r"who|where|which|would|will|did)\b.*\?\s*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """
You are a helpful, polite, and knowledgeable customer support assistant for a company that sells chargers
and other electronic devices.

Your job is to answer customer questions using only the information provided in the retrieved context
and any past user question and answer.

Do not invent information.

Your objectives:
- Provide clear, accurate, and concise responses.
- Ask clarifying questions if the user request is vague or missing key details.
- Stick strictly to known facts. Never speculate or make up information.
- If the request is outside your scope (e.g. refunds, legal issues), politely direct the user to human support.

For escalation or human support, refer customers to: support@chargepro.com

Tone guidelines:
- Be friendly, professional, and empathetic.
- Match the customer's tone, but never be sarcastic or emotional.
- Prioritize helpfulness, clarity, and honesty.

If the answer is not found in the context or previous interactions, respond with:
"Sorry, I cannot answer that based on the available information."
"""


def clean_qa_context(raw: str) -> str:
    """
    Transform a raw QA context string into a clean, structured list of Q&A:

    Args:
        raw: string containing Q&A pairs separated by '---'.

    Returns:
        A formatted string of "Q: ...\nA: ..." entries.
    """
    blocks = [blk.strip() for blk in raw.split("---") if blk.strip()]
    formatted: List[str] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        question = lines[0]
        answers = [ln for ln in lines[1:] if not QUESTION_PATTERN.match(ln)]
        if not answers:
            continue

        formatted.append(f"Q: {question}\nA: {' '.join(answers)}")

    return "\n".join(formatted)


def format_prompt(question: str, context: str = "") -> str:
    """
    Create the final prompt for the LLM, injecting cleaned context if available.

    Args:
        question: The user's question.
        context: Raw context string from RAG retrieval.

    Returns:
        A single string prompt for the model.
    """
    context_section = f"{clean_qa_context(context)}" if context else ""

    prompt_sections: List[str] = [
        "Given the context next and previous messages, reply to the user question or input. Be consice."
    ]
    prompt_sections.append("Context:")
    prompt_sections.append("---")
    if context_section:
        prompt_sections.append(context_section)
    prompt_sections.append("---")
    prompt_sections.append(f"User question: {question}")

    return "\n".join(prompt_sections)


def get_initial_chat_state() -> List[Dict[str, str]]:
    """
    Initialize the conversation with the system prompt.

    Returns:
        A list containing a single system message dictionary.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]
