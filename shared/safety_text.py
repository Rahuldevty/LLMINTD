from typing import Iterable

from shared.semantic_guard import assess_risk


_SAFE_REWRITE_FALLBACK = "Please restate the request in a safe, policy-compliant, and constructive form."


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def build_safe_rewrite(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return _SAFE_REWRITE_FALLBACK

    risk = assess_risk(prompt)
    categories = set(risk.get("categories", []))
    top_label = risk.get("top_unsafe_label", "none")

    if "self_harm" in categories or top_label == "self_harm":
        return (
            "Provide compassionate crisis support, encourage contacting local emergency services or a "
            "suicide prevention hotline, and suggest reaching out to a trusted person right away."
        )

    if "violence" in categories or "hate_abuse" in categories or top_label in {"violence", "hate_abuse"}:
        return (
            "Explain safe, non-violent ways to handle conflict, de-escalate the situation, set boundaries, "
            "and get help from appropriate local support if needed."
        )

    if "cyber_abuse" in categories or top_label == "cyber_abuse":
        return (
            "Explain ethical cybersecurity best practices for defending a website, securing accounts, and "
            "reporting vulnerabilities responsibly."
        )

    if "illegal_activity" in categories or top_label == "illegal_activity":
        return (
            "Suggest legal and ethical alternatives that solve the underlying problem without breaking the law "
            "or putting anyone at risk."
        )

    if "prompt_attack" in categories or top_label == "prompt_attack":
        return (
            "Explain that system instructions and safety controls should be respected, and provide safe help "
            "within policy-compliant boundaries."
        )

    return prompt


def build_safe_response(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return "Please share a question or task, and I’ll help with a safe, useful answer."

    risk = assess_risk(prompt)
    decision = risk.get("decision", "ALLOW")
    if decision != "ALLOW":
        rewritten = build_safe_rewrite(prompt)
        if "self_harm" in set(risk.get("categories", [])) or risk.get("top_unsafe_label") == "self_harm":
            return (
                "I can’t help with harming yourself. If this is urgent, contact local emergency services now "
                "or reach out to a suicide prevention hotline and a trusted person immediately."
            )
        return (
            "I can’t help with harmful or illegal instructions. "
            f"A safer direction would be: {rewritten}"
        )

    lowered = prompt.lower()
    if _contains_any(lowered, ["artificial intelligence", " ai", "what is ai", "what is artificial intelligence"]):
        return (
            "Artificial intelligence is the field of building computer systems that can learn from data, "
            "recognize patterns, reason about problems, and assist with tasks that normally need human intelligence."
        )
    if _contains_any(lowered, ["python programming", "what is python", "python language"]):
        return (
            "Python is a general-purpose programming language known for readable syntax and a large ecosystem. "
            "It is widely used for web development, automation, data science, and AI."
        )
    if _contains_any(lowered, ["machine learning"]):
        return (
            "Machine learning is a branch of AI where systems learn patterns from data to make predictions or "
            "decisions without being explicitly programmed for every case."
        )

    return (
        "Here is a safe high-level answer: clarify the goal, use reliable sources, and focus on practical next "
        "steps that are lawful, constructive, and low risk."
    )


def sanitize_generated_text(text: str, fallback: str) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return fallback

    trailing_fragments = (
        "however",
        "but",
        "therefore",
        "although",
        "though",
        "meanwhile",
    )
    if cleaned.lower().endswith(trailing_fragments):
        cleaned = cleaned.rsplit(" ", 1)[0].rstrip(",;:- ")

    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."

    return cleaned or fallback
