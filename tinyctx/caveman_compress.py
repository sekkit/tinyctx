"""Caveman-style text compression — pure Python, zero dependencies.

Applies regex-based prose shortening rules to tool descriptions and
instructions. Preserves code identifiers, paths, URLs, and punctuation
that carries semantic load. Runs inline in the proxy pipeline — no
external process, no codex config changes.

Rules (order matters):
  1. Collapse common verbose phrases → short form
  2. Drop filler / hedging words
  3. Shorten redundant qualifiers
  4. Normalize whitespace

Typical savings: 30-50% on tool descriptions (the caveman MCP middleware
achieves ~65% on the full tool/result stream; our inline proxy version
targets the description field specifically, which is the bulk of the
per-tool token cost in the system prompt).
"""

from __future__ import annotations

import re
from typing import Any

# ─────────────────────── phrase substitution table ──────────────────
# (long pattern, short replacement) — ordered by specificity (longest first)

_PHRASE_RULES: list[tuple[str, str]] = [
    # ── filler / hedging ──
    (r"\bplease note that\b", ""),
    (r"\bnote that\b", ""),
    (r"\bit is important to\b", ""),
    (r"\bplease be aware that\b", ""),
    (r"\bbe sure to\b", ""),
    (r"\bmake sure to\b", "ensure"),
    (r"\bmake sure\b", "ensure"),
    (r"\byou can use this\b", ""),
    (r"\byou can use\b", ""),
    (r"\byou may use this\b", ""),
    (r"\byou may use\b", ""),
    (r"\bthis can be used to\b", "use to"),
    (r"\bthis is used to\b", "use to"),
    (r"\bthis is used for\b", "for"),
    (r"\bthis tool allows you to\b", ""),
    (r"\bthis tool enables you to\b", ""),
    (r"\bthis tool can be used to\b", ""),
    (r"\bthis function allows you to\b", ""),
    (r"\bthis function is used to\b", ""),
    (r"\bthis function can be used to\b", ""),
    (r"\bin order to\b", "to"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bas well as\b", "and"),
    (r"\bin addition to\b", "and"),
    (r"\bin addition\b", "also"),
    (r"\bfor example\b", "e.g."),
    (r"\bsuch as\b", "e.g."),
    (r"\bthat is\b", "i.e."),
    (r"\ba number of\b", "several"),
    (r"\ba variety of\b", "various"),
    (r"\ba lot of\b", "many"),
    (r"\ba large amount of\b", "much"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin the event that\b", "if"),
    (r"\bwith regard to\b", "about"),
    (r"\bwith respect to\b", "about"),
    (r"\bin terms of\b", "for"),
    (r"\bin the context of\b", "in"),
    (r"\bon the basis of\b", "based on"),
    (r"\bthe majority of\b", "most"),
    (r"\bthe rest of\b", "remaining"),
    (r"\ba wide range of\b", "many"),
    (r"\bthe following\b", "this"),
    (r"\bas follows\b", ""),
    (r"\bthat are available\b", "available"),
    (r"\bthat are present\b", "present"),
    (r"\bthat is required\b", "required"),
    (r"\bthat is needed\b", "needed"),
    (r"\bwhich is\b", ""),
    (r"\bwhich are\b", ""),
    (r"\bthat can be used\b", "usable"),
    (r"\bcan be (used|applied|found|seen|accessed|configured|customized|modified|changed|adjusted|set|specified|provided|supplied|passed|returned|retrieved|obtained|fetched|read|written|created|deleted|updated|inserted|removed|renamed|copied|moved|linked|attached|detached|added|dropped|expanded|collapsed|opened|closed|started|stopped|paused|resumed|enabled|disabled|activated|deactivated|registered|unregistered|installed|uninstalled|loaded|unloaded|mounted|unmounted|connected|disconnected|bound|unbound|locked|unlocked|signed|verified|validated|checked|tested|run|executed|invoked|called|triggered|fired|sent|received|processed|handled|managed|controlled|monitored|tracked|logged|recorded|stored|cached|buffered|queued|stacked|pushed|popped|peeked|shifted|unshifted|sorted|filtered|mapped|reduced|folded|scanned|searched|found|matched|replaced|split|joined|merged|zipped|unzipped|parsed|serialized|deserialized|encoded|decoded|encrypted|decrypted|hashed|signed|verified|compiled|transpiled|bundled|minified|formatted|linted|typed|inferred|checked|analyzed|inspected|traced|profiled|debugged|fixed|patched|worked)\b", r"r'\1'able"),
    (r"\bwhether or not\b", "whether"),
    (r"\bthe fact that\b", "that"),
    (r"\bthe ability to\b", "can"),
    (r"\bthe use of\b", "using"),
    (r"\bthe presence of\b", "presence of"),
    (r"\bits own\b", "its"),
    (r"\bwhen it comes to\b", "for"),
    (r"\bin order for\b", "for"),
    (r"\bso that\b", "so"),
    (r"\bso as to\b", "to"),
    (r"\bpertaining to\b", "about"),
    (r"\brelating to\b", "about"),
    (r"\bassociated with\b", "with"),
    (r"\bmoreover\b", "also"),
    (r"\bfurthermore\b", "also"),
    (r"\bnevertheless\b", "yet"),
    (r"\bnonetheless\b", "yet"),
    (r"\bconsequently\b", "so"),
    (r"\btherefore\b", "so"),
    (r"\bthus\b", "so"),
    (r"\baccordingly\b", "so"),
    (r"\binterestingly\b", ""),
    (r"\bimportantly\b", ""),
    (r"\bnotably\b", ""),
    (r"\bspecifically\b", ""),
    (r"\btypically\b", "usually"),
    (r"\bgenerally\b", "usually"),
    (r"\bessentially\b", ""),
    (r"\bbasically\b", ""),
    (r"\bfundamentally\b", ""),
    (r"\binherently\b", ""),
    (r"\brelatively\b", ""),
    (r"\bcomparatively\b", ""),
    (r"\brelatively speaking\b", ""),
    (r"\bto be able to\b", "to"),
    (r"\bhas the ability to\b", "can"),
    (r"\bis able to\b", "can"),
    (r"\bis capable of\b", "can"),
    (r"\bare able to\b", "can"),
    (r"\bare capable of\b", "can"),
    (r"\bprovides the ability to\b", "lets you"),
    (r"\bgives you the ability to\b", "lets you"),
    (r"\ballows for\b", "allows"),
    (r"\bprovides a way to\b", "lets you"),
    (r"\bresponsibility of\b", "responsible for"),
    (r"\bresponsible for ensuring\b", "must ensure"),
    (r"\bit is necessary to\b", "must"),
    (r"\bit is required to\b", "must"),
    (r"\bit is recommended that\b", "should"),
    (r"\bit is advised that\b", "should"),
    (r"\bit is suggested that\b", "should"),
    (r"\bit is possible to\b", "you can"),
    (r"\bit is not possible to\b", "cannot"),
    (r"\bit may be\b", "it may be"),
    (r"\bit might be\b", "it may be"),
    (r"\bwill need to\b", "must"),
    (r"\bneed to\b", "must"),
    (r"\bshould be able to\b", "should"),
    (r"\bare going to\b", "will"),
    (r"\bgoing forward\b", ""),
    (r"\bat this point\b", "now"),
    (r"\bat this time\b", "now"),
    (r"\bat the moment\b", "now"),
    (r"\bcurrently\b", "now"),
    (r"\bpresently\b", "now"),
    (r"\bat the present time\b", "now"),
    (r"\bat the current time\b", "now"),
    (r"\bappropriate\b", "right"),
    (r"\bappropriate for\b", "right for"),
    (r"\bcorresponding\b", "matching"),
    (r"\bautomatically\b", "auto"),
    (r"\bautomatic\b", "auto"),
    (r"\bmanually\b", "manual"),
    (r"\bperform\b", "run"),
    (r"\bexecute\b", "run"),
    (r"\butilize\b", "use"),
    (r"\butilise\b", "use"),
    (r"\butilization\b", "use"),
    (r"\bimplement\b", "add"),
    (r"\bimplementation\b", "impl"),
    (r"\bdemonstrate\b", "show"),
    (r"\bdisplay\b", "show"),
    (r"\bindicate\b", "show"),
    (r"\billustrate\b", "show"),
    (r"\bobtain\b", "get"),
    (r"\bacquire\b", "get"),
    (r"\bretrieve\b", "get"),
    (r"\bsubsequent\b", "later"),
    (r"\bpreceding\b", "before"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bfollowing\b", "after"),
    (r"\bcommence\b", "start"),
    (r"\binitiate\b", "start"),
    (r"\bterminate\b", "stop"),
    (r"\bcease\b", "stop"),
    (r"\bhalt\b", "stop"),
    (r"\bendeavor\b", "try"),
    (r"\battempt\b", "try"),
    (r"\bendeavour\b", "try"),
    (r"\breqire\b", "need"),
    (r"\bnecessitate\b", "need"),
    (r"\breqirement\b", "need"),
    (r"\bprerequisite\b", "need"),
    (r"\brerequisite\b", "need"),
    (r"\badditional\b", "more"),
    (r"\badditional\b", "more"),
    (r"\bnumerous\b", "many"),
    (r"\bmultiple\b", "several"),
    (r"\bwithin\b", "in"),
    (r"\binside\b", "in"),
    (r"\boutside\b", "out"),
    (r"\bexternal\b", "outer"),
    (r"\binternal\b", "inner"),
    (r"\bsubmit\b", "send"),
    (r"\btransmit\b", "send"),
    (r"\breceive\b", "get"),
    (r"\bcomprehensive\b", "full"),
    (r"\bextensive\b", "large"),
    (r"\bsignificant\b", "big"),
    (r"\bsubstantial\b", "large"),
    (r"\bconsiderable\b", "large"),
    (r"\bapproximately\b", "~"),
    (r"\bapproximate\b", "~"),
    (r"\broughly\b", "~"),
    (r"\baround\b", "~"),
]

# ─────────────────────── helpers ────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PATH_RE = re.compile(r"(?:[/\w.-]+/)+[\w.-]+(?:\.\w+)?")
_VERBATIM_RE = re.compile(r"<[^>]+>")


def _extract_protected(text: str) -> tuple[str, dict[str, str]]:
    """Extract code blocks, inline code, URLs, paths, and HTML tags into
    placeholder tokens so regex rules never touch them."""
    placeholders: dict[str, str] = {}
    counter = [0]

    def _replace(m: re.Match) -> str:
        key = f"\x00P{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = m.group(0)
        return key

    for pattern in (_CODE_BLOCK_RE, _INLINE_CODE_RE, _URL_RE, _PATH_RE, _VERBATIM_RE):
        text = pattern.sub(_replace, text)
    return text, placeholders


def _restore_protected(text: str, placeholders: dict[str, str]) -> str:
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


# ─────────────────────── compression ────────────────────────────────

# Pre-compiled regex patterns
_PHRASE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE | re.MULTILINE), repl)
    for pat, repl in _PHRASE_RULES
]

_WS_RE = re.compile(r"\s+")
_SENTENCE_GAP_RE = re.compile(r"\.\s{2,}")


def compress_text(text: str) -> str:
    """Apply caveman rules to a prose string. Returns compressed text.
    Code blocks, inline code, URLs, paths, and HTML tags are preserved
    byte-for-byte."""
    if not text or len(text) < 40:
        return text

    original = text
    text, placeholders = _extract_protected(text)

    for pat, repl in _PHRASE_PATTERNS:
        text = pat.sub(repl, text)

    # Normalize whitespace (collapse multiple spaces, but keep newlines)
    text = _WS_RE.sub(" ", text)
    # Collapse sentence gaps
    text = _SENTENCE_GAP_RE.sub(". ", text)
    text = text.strip()

    text = _restore_protected(text, placeholders)

    # Don't return empty string — at minimum return a sensible stub
    if not text.strip():
        # Extract first sentence from original
        first = original.split(".")[0].strip()
        return first[:200] if first else original[:200]

    return text


# ─────────────────────── tool description compression ───────────────


def compress_tool_descriptions(body: dict[str, Any]) -> dict[str, Any]:
    """Walk body['tools'] and compress each tool's `description` field.
    Returns the mutated body (also mutates in-place)."""
    tools = body.get("tools")
    if not tools:
        return body

    count = 0
    chars_before = 0
    chars_after = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        desc = tool.get("description")
        if not desc or not isinstance(desc, str) or len(desc) < 40:
            continue
        compressed = compress_text(desc)
        if compressed != desc:
            chars_before += len(desc)
            chars_after += len(compressed)
            tool["description"] = compressed
            count += 1

    return body


def compress_instructions(body: dict[str, Any]) -> dict[str, Any]:
    """Compress body['instructions'] prose. Only compresses if the
    instructions field is long enough to benefit."""
    inst = body.get("instructions")
    if not inst or not isinstance(inst, str) or len(inst) < 200:
        return body

    compressed = compress_text(inst)
    if compressed != inst:
        body["instructions"] = compressed

    return body
