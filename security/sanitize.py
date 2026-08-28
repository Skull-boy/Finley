"""
Input sanitization & prompt-injection detection.

OWASP A03 (Injection) + LLM01 (Prompt Injection) + LLM02 (Insecure Output).

* sanitize_input — strips control characters before storage or LLM call.
* detect_prompt_injection — flags attempts to override system instructions.
* sanitize_html_for_telegram — safe fallback (used by agent formatter, but
  exposed here for reuse).

No new dependencies — pure stdlib + re.
"""
import re
import unicodedata

# ─── Control-char stripping ────────────────────────────────────────────────

# Keep printable + common whitespace (\n, \t). Drop other C0/C1 controls,
# zero-width joiners/bidi overrides that hide injection.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060-\u2064\u202a-\u202e]")


def sanitize_input(text: str, max_len: int = 4000) -> str:
    """Normalize, strip invisible controls, and truncate safely."""
    if not text or not isinstance(text, str):
        return ""
    # NFC normalizes visually identical sequences (prevents bypass via combining marks)
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    # Collapse excessive whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]{4,}", " ", text)
    # Truncate on char boundary — caller already enforces 4000
    if len(text) > max_len:
        text = text[:max_len]
    return text.strip()


# ─── Prompt-injection detection ───────────────────────────────────────────

# Common jailbreak / system-override phrases (case-insensitive).
# Kept small and high-precision to avoid false positives on finance chatter.
# If you broaden this, keep it precision-first — false positives block real users.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(prior|previous|system)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"system\s*:\s*",
    r"jailbreak",
    r"DAN\s+mode",
    r"do\s+anything\s+now",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"repeat\s+(your\s+)?(system\s+)?instructions",
    r"output\s+your\s+initial\s+prompt",
    r"<\s*/?\s*system\s*>",
    r"\[SYSTEM\]",
    r"```system",
]

_COMPILED_INJECTION = re.compile(
    "|".join(f"(?:{p})" for p in _INJECTION_PATTERNS),
    re.IGNORECASE,
)


def detect_prompt_injection(text: str) -> bool:
    """
    Heuristic: True if text looks like an attempt to override Finley's role.

    This is a *guardrail*, not a proof. Agent must still treat user content
    as untrusted data (never as system instruction) regardless of this flag.
    """
    if not text or len(text) < 10:
        return False
    return bool(_COMPILED_INJECTION.search(text))


# ─── HTML safety ───────────────────────────────────────────────────────────

_ALLOWED_TAGS = {"b", "i", "code", "a"}

_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")


def strip_disallowed_html(text: str) -> str:
    """Remove HTML tags not in the Telegram allowlist (b/i/code/a)."""
    def _repl(m: re.Match) -> str:
        tag = m.group(1).lower()
        if tag in _ALLOWED_TAGS:
            return m.group(0)
        return ""
    return _TAG_RE.sub(_repl, text)
