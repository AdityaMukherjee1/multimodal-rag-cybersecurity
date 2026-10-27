"""Lightweight query intent router for pre-retrieval classification.

Classifies user queries into interaction modes:
  - CONVERSATIONAL: Greetings, thanks, acknowledgements, casual chat (no active evidence).
  - CYBERSECURITY_KNOWLEDGE: General cybersecurity concepts, CVEs, techniques, explanatory questions.
  - REMEDIATION: Remediation advice ("what should I do to fix these?", mitigation, containment).
  - INVESTIGATION: Incident investigation, log analysis, threat hunting, attack attribution.

When uploaded evidence or active session evidence is present:
  - A query like "hi" or "hello" is NOT treated as pure chitchat; it is routed to analyze
    and summarize the uploaded evidence.
  - "what this says" is routed to inspect and explain the artifact.
  - "what should i do to fix these?" is routed to remediation using the evidence context.

Rule-based, deterministic, zero-GPU, <1ms classification latency.
Does NOT modify the RAG, multimodal, verification, or research architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Classification intent categories."""

    CONVERSATIONAL = "CONVERSATIONAL"
    CYBERSECURITY_KNOWLEDGE = "CYBERSECURITY_KNOWLEDGE"
    REMEDIATION = "REMEDIATION"
    INVESTIGATION = "INVESTIGATION"


@dataclass
class IntentClassification:
    """Result of query intent classification.

    Attributes:
        intent: The classified intent category.
        confidence: Classification confidence (0.0–1.0).
        reason: Human-readable explanation of the classification decision.
    """

    intent: QueryIntent
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Pattern Banks (compiled once at module import)
# ---------------------------------------------------------------------------

# Greeting patterns — single words or short phrases
_GREETING_TOKENS = frozenset({
    "hi", "hii", "hiii", "hello", "hey", "heyy", "heya", "howdy",
    "yo", "sup", "wassup", "whatsup",
    "hola", "namaste", "greetings",
})

_FAREWELL_TOKENS = frozenset({
    "bye", "goodbye", "cya", "later", "goodnight",
})

_THANKS_TOKENS = frozenset({
    "thanks", "thankyou", "thx", "ty", "tysm", "appreciated",
    "thank", "cheers",
})

_ACKNOWLEDGEMENT_TOKENS = frozenset({
    "ok", "okay", "sure", "alright", "cool", "nice", "great",
    "awesome", "perfect", "fine", "got", "understood", "noted",
    "yep", "yup", "yeah", "yes", "nope", "no", "nah",
})

_SOCIAL_PHRASES = [
    re.compile(r"^good\s+(morning|afternoon|evening|night|day)$", re.I),
    re.compile(r"^how\s+are\s+you", re.I),
    re.compile(r"^what'?s?\s+up", re.I),
    re.compile(r"^thank\s+you", re.I),
    re.compile(r"^nice\s+to\s+meet", re.I),
    re.compile(r"^who\s+are\s+you", re.I),
    re.compile(r"^what\s+can\s+you\s+do", re.I),
    re.compile(r"^what\s+are\s+you", re.I),
    re.compile(r"^help$", re.I),
]

# Remediation / "how to fix" patterns
_REMEDIATION_PATTERNS = re.compile(
    r"\b("
    r"how\s+to\s+fix|what\s+should\s+i\s+do\s+to\s+fix|how\s+do\s+i\s+fix|"
    r"what\s+should\s+i\s+do|how\s+to\s+remediate|remediation|mitigate|"
    r"how\s+to\s+mitigate|mitigation|patch\s+this|how\s+to\s+patch|"
    r"how\s+to\s+contain|containment|incident\s+response\s+steps|"
    r"how\s+to\s+clean|next\s+steps|how\s+to\s+protect|how\s+to\s+prevent|"
    r"countermeasures?|remedy|remedies|fix\s+these|fix\s+this"
    r")\b",
    re.I,
)

# Content explanation / "what this says" patterns
_CONTENT_EXPLANATION_PATTERNS = re.compile(
    r"\b("
    r"what\s+this\s+says|what\s+does\s+this\s+say|what\s+is\s+this|"
    r"explain\s+this\s+(image|screenshot|file|log|report|output|document|error)|"
    r"what\s+is\s+written\s+here|read\s+this|summarize\s+this|"
    r"what\s+happened\s+here|what\s+does\s+this\s+mean|"
    r"what\s+does\s+this\s+command\s+do|what\s+is\s+this\s+command|"
    r"explain\s+this\s+command"
    r")\b",
    re.I,
)

# Deep incident investigation verbs & phrases
_INVESTIGATION_VERBS = re.compile(
    r"\b("
    r"investigate|analyze\s+this\s+incident|investigate\s+these\s+logs|"
    r"find\s+the\s+attack|correlate\s+these\s+events|why\s+did\s+this\s+compromise|"
    r"threat\s+hunt|forensic\s+analysis|trace\s+adversary|root\s+cause|"
    r"attribution|lateral\s+movement\s+investigation|privilege\s+escalation\s+path|"
    r"inspect\s+telemetry|examine\s+evtx|triage\s+alert|audit\s+indicators"
    r")\b",
    re.I,
)

# Cybersecurity general keywords
_CYBER_KEYWORDS = re.compile(
    r"\b("
    r"att&ck|att\&ck|mitre|attack|technique|tactic|"
    r"cve[-‑]?\d{4}|cve|cwe[-‑]?\d+|vulnerability|exploit|"
    r"phishing|malware|ransomware|trojan|worm|botnet|rootkit|backdoor|"
    r"incident|breach|intrusion|compromise|lateral\s+movement|"
    r"powershell|mimikatz|cobalt\s*strike|empire|metasploit|bloodhound|kerberoast(ing)?|"
    r"firewall|ids|ips|siem|edr|xdr|ndr|"
    r"log|logs|evtx|pcap|packet|flow|"
    r"ioc|indicator|hash|md5|sha256|sha1|"
    r"encryption|decryption|certificate|tls|ssl|"
    r"privilege\s+escalation|credential|brute\s*force|pass[- ]the[- ]hash|"
    r"c2|command\s+and\s+control|exfiltration|"
    r"network|port\s+scan|reconnaissance|ddos|dos|denial\s+of\s+service|"
    r"sqli|xss|csrf|rce|ssrf|buffer\s*overflow|"
    r"stuxnet|wannacry|notpetya|solarwinds|heartbleed|eternalblue|emotet|trickbot|darkside|lockbit|"
    r"wireshark|nmap|burp|nessus|"
    r"cyber|security|threat|adversary|apt|"
    r"registry|persistence|injection|"
    r"detection|alert|rule|sigma|yara|snort|"
    r"dns|http|tcp|udp|ip\s+address|domain|url|"
    r"authentication|authorization|access\s+control|"
    r"patch|update|zero[\s-]?day|"
    r"forensic|evidence|artifact|screenshot|capture"
    r")\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Conversational response templates
# ---------------------------------------------------------------------------

_GREETING_RESPONSES = [
    (
        "Hello! I'm your **CyberAI Investigator** — an evidence-grounded cybersecurity assistant.\n\n"
        "I can help you:\n"
        "- 🔍 Investigate cybersecurity incidents, logs, and artifacts\n"
        "- 🛡️ Explain MITRE ATT&CK techniques, CVE vulnerabilities, and threats\n"
        "- 📊 Analyze uploaded logs, screenshots, and forensic reports\n"
        "- 🔧 Recommend practical remediation and mitigation steps\n"
        "- 🔗 Cross-correlate evidence across knowledge bases with claim verification\n\n"
        "How can I help with your cybersecurity questions or investigation today?"
    ),
]

_THANKS_RESPONSES = [
    (
        "You're welcome! If you have any further questions or forensic evidence to review, "
        "feel free to ask anytime. 🛡️"
    ),
]

_FAREWELL_RESPONSES = [
    "Goodbye! Stay secure. 🔒 Feel free to return anytime you need cybersecurity analysis.",
]

_ACKNOWLEDGEMENT_RESPONSES = [
    "Understood! Let me know whenever you'd like to investigate an incident or ask a cybersecurity question. 🛡️",
]

_IDENTITY_RESPONSES = [
    (
        "I'm **CyberAI Investigator**, an AI-powered assistant designed for evidence-grounded cybersecurity incident investigation.\n\n"
        "Under the hood, I combine:\n"
        "- 📄 **Dense Text Retrieval** across MITRE ATT&CK, NVD CVEs, and Mordor/EVTX datasets\n"
        "- 🖼️ **Multimodal Retrieval** (CLIP + OCR) for terminal screenshots and visual artifacts\n"
        "- 🧠 **AI Reasoning** that answers general questions or conducts formal forensic investigations\n"
        "- 🛡️ **Claim-Level Evidence Verification** with epistemic auditing to guarantee answers are grounded in real data\n\n"
        "You can ask me questions naturally, or attach forensic logs, screenshots, and reports to investigate!"
    ),
]


class QueryIntentRouter:
    """Lightweight rule-based query intent classifier.

    Distinguishes:
      - CONVERSATIONAL: Pure chat, no active evidence.
      - REMEDIATION: Remediation and mitigation advice.
      - CYBERSECURITY_KNOWLEDGE: General cybersecurity questions and content explanations.
      - INVESTIGATION: Incident investigations, log correlation, and threat analysis.

    Deterministic pattern matching, zero GPU, <1ms latency.
    """

    def classify(
        self,
        query: str,
        has_file_attachment: bool = False,
        session_has_evidence: bool = False,
    ) -> IntentClassification:
        """Classify the intent of a user query.

        Args:
            query: The raw user query string.
            has_file_attachment: Whether a file was attached to the current query.
            session_has_evidence: Whether previous turns in the session contain evidence.

        Returns:
            IntentClassification with intent, confidence, and reason.
        """
        evidence_present = has_file_attachment or session_has_evidence
        q_stripped = query.strip()
        q_lower = q_stripped.lower()
        q_clean = re.sub(r"[^\w\s]", "", q_lower).strip()
        tokens = q_clean.split()
        word_count = len(tokens)

        # ------------------------------------------------------------------
        # SCENARIO A: EVIDENCE IS PRESENT (Attached or in Active Session)
        # ------------------------------------------------------------------
        if evidence_present:
            # Check 1: User asks remediation about the evidence
            if _REMEDIATION_PATTERNS.search(q_stripped):
                return IntentClassification(
                    intent=QueryIntent.REMEDIATION,
                    confidence=0.95,
                    reason="Remediation guidance requested for active evidence context.",
                )

            # Check 2: User asks to explain what the evidence/file says
            if _CONTENT_EXPLANATION_PATTERNS.search(q_stripped):
                return IntentClassification(
                    intent=QueryIntent.CYBERSECURITY_KNOWLEDGE,
                    confidence=0.95,
                    reason="Content explanation requested for attached/session evidence.",
                )

            # Check 3: Formal investigation query
            if _INVESTIGATION_VERBS.search(q_stripped):
                return IntentClassification(
                    intent=QueryIntent.INVESTIGATION,
                    confidence=0.95,
                    reason="Formal incident investigation requested for active evidence.",
                )

            # Check 4: User sent a greeting (e.g. "hi", "hello") WITH an uploaded file
            # The uploaded file changes context! Do NOT treat as pure chitchat ignoring the file.
            is_greeting = (
                word_count == 0
                or (word_count == 1 and tokens[0] in _GREETING_TOKENS)
                or any(p.search(q_lower) for p in _SOCIAL_PHRASES)
            )
            if is_greeting:
                return IntentClassification(
                    intent=QueryIntent.INVESTIGATION,
                    confidence=0.95,
                    reason="File attached with greeting — analyzing uploaded evidence and providing overview.",
                )

            # Check 5: General cyber question while evidence is loaded
            if _CYBER_KEYWORDS.search(q_stripped):
                return IntentClassification(
                    intent=QueryIntent.CYBERSECURITY_KNOWLEDGE,
                    confidence=0.90,
                    reason="Cybersecurity query evaluated against active evidence and knowledge base.",
                )

            # Default with evidence: treat as investigation/analysis of the evidence
            return IntentClassification(
                intent=QueryIntent.INVESTIGATION,
                confidence=0.90,
                reason="Active evidence present — binding query to investigation context.",
            )

        # ------------------------------------------------------------------
        # SCENARIO B: NO EVIDENCE PRESENT (Standard Text Conversation)
        # ------------------------------------------------------------------

        # Empty query without evidence
        if not q_stripped:
            return IntentClassification(
                intent=QueryIntent.CONVERSATIONAL,
                confidence=1.0,
                reason="Empty query string.",
            )

        # Check 1: Remediation without prior evidence
        if _REMEDIATION_PATTERNS.search(q_stripped):
            return IntentClassification(
                intent=QueryIntent.REMEDIATION,
                confidence=0.90,
                reason="General cybersecurity remediation guidance requested.",
            )

        # Check 2: Content explanation without attached evidence
        if _CONTENT_EXPLANATION_PATTERNS.search(q_stripped):
            return IntentClassification(
                intent=QueryIntent.CYBERSECURITY_KNOWLEDGE,
                confidence=0.85,
                reason="Content explanation requested.",
            )

        # Check 3: Short queries matched against conversational token sets
        if word_count <= 4:
            # Single-token matches
            if word_count == 1:
                token = tokens[0]
                if token in _GREETING_TOKENS:
                    return IntentClassification(
                        intent=QueryIntent.CONVERSATIONAL,
                        confidence=0.98,
                        reason=f"Greeting detected: '{token}'.",
                    )
                if token in _FAREWELL_TOKENS:
                    return IntentClassification(
                        intent=QueryIntent.CONVERSATIONAL,
                        confidence=0.97,
                        reason=f"Farewell detected: '{token}'.",
                    )
                if token in _THANKS_TOKENS:
                    return IntentClassification(
                        intent=QueryIntent.CONVERSATIONAL,
                        confidence=0.98,
                        reason=f"Thanks/acknowledgement detected: '{token}'.",
                    )
                if token in _ACKNOWLEDGEMENT_TOKENS:
                    return IntentClassification(
                        intent=QueryIntent.CONVERSATIONAL,
                        confidence=0.95,
                        reason=f"Acknowledgement detected: '{token}'.",
                    )

            # Multi-token social phrase matching
            for pattern in _SOCIAL_PHRASES:
                if pattern.search(q_lower):
                    return IntentClassification(
                        intent=QueryIntent.CONVERSATIONAL,
                        confidence=0.95,
                        reason=f"Social phrase detected: '{q_stripped}'.",
                    )

            # Short query containing cyber keywords → NOT conversational
            if _CYBER_KEYWORDS.search(q_stripped):
                if _INVESTIGATION_VERBS.search(q_stripped):
                    return IntentClassification(
                        intent=QueryIntent.INVESTIGATION,
                        confidence=0.85,
                        reason="Short query with investigation verb and cybersecurity keywords.",
                    )
                return IntentClassification(
                    intent=QueryIntent.CYBERSECURITY_KNOWLEDGE,
                    confidence=0.85,
                    reason="Short query with cybersecurity keywords.",
                )

            # Very short query with recognized conversational tokens → conversational
            conversational_words = _GREETING_TOKENS | _FAREWELL_TOKENS | _THANKS_TOKENS | _ACKNOWLEDGEMENT_TOKENS
            if word_count <= 2 and any(t in conversational_words for t in tokens):
                return IntentClassification(
                    intent=QueryIntent.CONVERSATIONAL,
                    confidence=0.85,
                    reason=f"Conversational token detected in short input: '{q_stripped}'.",
                )

        # Check 4: Longer queries
        has_investigation = bool(_INVESTIGATION_VERBS.search(q_stripped))
        has_cyber = bool(_CYBER_KEYWORDS.search(q_stripped))

        if has_investigation:
            return IntentClassification(
                intent=QueryIntent.INVESTIGATION,
                confidence=0.90 if has_cyber else 0.80,
                reason="Investigation verb detected.",
            )

        if has_cyber:
            return IntentClassification(
                intent=QueryIntent.CYBERSECURITY_KNOWLEDGE,
                confidence=0.85,
                reason="Cybersecurity keywords detected.",
            )

        # Check social phrases in longer queries
        for pattern in _SOCIAL_PHRASES:
            if pattern.search(q_lower):
                return IntentClassification(
                    intent=QueryIntent.CONVERSATIONAL,
                    confidence=0.85,
                    reason=f"Social phrase detected: '{q_stripped}'.",
                )

        # Default: treat ambiguous queries as knowledge questions (run RAG)
        return IntentClassification(
            intent=QueryIntent.CYBERSECURITY_KNOWLEDGE,
            confidence=0.60,
            reason="Ambiguous query — defaulting to knowledge retrieval.",
        )

    def get_conversational_response(self, query: str) -> str:
        """Generate an appropriate conversational response for a non-investigation query.

        Args:
            query: The original user query.

        Returns:
            A friendly, contextual response string.
        """
        q_lower = query.strip().lower()
        q_clean = re.sub(r"[^\w\s]", "", q_lower).strip()
        tokens = q_clean.split()
        first_token = tokens[0] if tokens else ""

        # Identity questions
        for pattern in _SOCIAL_PHRASES:
            if pattern.search(q_lower):
                phrase_str = q_lower.strip()
                if "who are you" in phrase_str or "what can you do" in phrase_str or "what are you" in phrase_str:
                    return _IDENTITY_RESPONSES[0]
                if "how are you" in phrase_str:
                    return (
                        "I'm running smoothly and ready to assist! 🛡️ "
                        "Do you have a cybersecurity question or an incident you'd like to investigate?"
                    )

        # Greetings
        if first_token in _GREETING_TOKENS:
            return _GREETING_RESPONSES[0]

        # Thanks
        if first_token in _THANKS_TOKENS or "thank you" in q_lower:
            return _THANKS_RESPONSES[0]

        # Farewells
        if first_token in _FAREWELL_TOKENS:
            return _FAREWELL_RESPONSES[0]

        # Acknowledgements
        if first_token in _ACKNOWLEDGEMENT_TOKENS:
            return _ACKNOWLEDGEMENT_RESPONSES[0]

        # Generic fallback
        return _GREETING_RESPONSES[0]
