"""
LLM-based scam classifier, built ONLY for comparison against the project's
real, shipped model (TF-IDF + Logistic Regression -- see scam_detector.py).

This module is NOT used by the Flask app (app.py). It exists so
benchmark_llm_vs_baseline.py can run a fair, side-by-side comparison and
produce real numbers for the writeup, instead of guessing whether "a
frontier LLM would probably do better."

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   (your own key -- get one at
                                            console.anthropic.com; this is
                                            NOT free, see pricing notes in
                                            benchmark_llm_vs_baseline.py)

Uses Claude Haiku (the fastest/cheapest current Claude model) via forced
tool-use, so the response is always structured JSON rather than
free-form text we'd have to hope parses correctly.
"""
import os
import time

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

CLASSIFY_TOOL = {
    "name": "classify_scam_message",
    "description": (
        "Classify whether a text/email message is a scam targeting an "
        "elderly person, and explain why."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_scam": {
                "type": "boolean",
                "description": "True if this message is a scam, phishing attempt, or fraud targeting the recipient.",
            },
            "risk_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "low = safe, medium = some warning signs but unclear, high = clearly a scam.",
            },
            "scam_type": {
                "type": ["string", "null"],
                "description": (
                    "Short category name if this is a scam (e.g. 'grandparent_scam', "
                    "'government_impersonation', 'tech_support_scam', 'romance_scam', "
                    "'investment_fraud', 'business_email_compromise', 'lottery_sweepstakes', "
                    "'phishing_spoofing', 'package_delivery_scam', or another short label if "
                    "none of these fit). Null if not a scam."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One or two plain-language sentences a non-technical senior could understand, explaining the verdict.",
            },
        },
        "required": ["is_scam", "risk_level", "reasoning"],
    },
}

SYSTEM_PROMPT = (
    "You help protect elderly people from scam text messages and emails. "
    "You will be shown a single message (and optionally its sender address/number). "
    "Decide whether it is a scam, phishing attempt, or fraud attempt, and call the "
    "classify_scam_message tool with your answer. Be careful not to over-flag ordinary "
    "formal notifications (appointment reminders, delivery notices, bills) just because "
    "they are formal in tone -- only flag genuine warning signs (urgency, payment demands, "
    "requests for gift cards/wire transfers/remote access, threats, too-good-to-be-true offers, "
    "requests for personal/financial info, impersonation of an agency or relative)."
)


def classify_message_with_llm(client, text, sender=None, model=DEFAULT_MODEL, max_retries=3):
    """Classify one message with the LLM. Returns a dict with keys:
    label (0/1), risk_level, scam_type, reasoning, latency_sec, usage (dict), error (str or None).
    Retries on transient API errors with exponential backoff.
    """
    user_content = f"Message: {text}"
    if sender:
        user_content += f"\nSender: {sender}"

    last_err = None
    for attempt in range(max_retries):
        start = time.time()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_scam_message"},
                messages=[{"role": "user", "content": user_content}],
            )
            latency = time.time() - start

            tool_block = next(
                (b for b in resp.content if getattr(b, "type", None) == "tool_use"), None
            )
            if tool_block is None:
                raise ValueError("No tool_use block in LLM response")

            result = tool_block.input
            return {
                "label": 1 if result.get("is_scam") else 0,
                "risk_level": result.get("risk_level"),
                "scam_type": result.get("scam_type"),
                "reasoning": result.get("reasoning"),
                "latency_sec": latency,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 -- want to catch/report any SDK/network error
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff

    return {
        "label": None,
        "risk_level": None,
        "scam_type": None,
        "reasoning": None,
        "latency_sec": None,
        "usage": None,
        "error": last_err,
    }


def get_client():
    """Lazily import + construct the Anthropic client so this module can be
    imported (e.g. for its constants) without the `anthropic` package
    installed, unless you actually call this."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Get a key at console.anthropic.com "
            "and run: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    from anthropic import Anthropic
    return Anthropic()
