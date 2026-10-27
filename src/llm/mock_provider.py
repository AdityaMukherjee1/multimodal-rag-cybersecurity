"""Deterministic Mock LLM Provider for zero-cost reproducible testing.

Generates schema-compliant structured investigation answers referencing
the actual evidence IDs present in the retrieved context without requiring
network access or API keys.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from src.llm.base_llm import BaseLLM
from src.llm.output_parser import InvestigationOutputParser
from src.llm.schema import Claim, LLMResponse, StructuredInvestigationAnswer

logger = logging.getLogger(__name__)


class MockLLMProvider(BaseLLM):
    """Deterministic mock provider for automated unit and integration tests.

    Produces realistic, evidence-grounded structured JSON responses referencing
    the exact evidence IDs supplied in the context.
    """

    def __init__(
        self,
        model_name: str = "mock-investigator-v1",
        simulate_error: bool = False,
        error_message: str = "Mock simulated connection timeout",
        simulate_malformed_json: bool = False,
        simulate_plain_text: bool = False,
        response_delay_sec: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.simulate_error = simulate_error
        self.error_message = error_message
        self.simulate_malformed_json = simulate_malformed_json
        self.simulate_plain_text = simulate_plain_text
        self.response_delay_sec = response_delay_sec

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate mock response string."""
        if self.simulate_error:
            logger.error("MockLLMProvider raising simulated error: %s", self.error_message)
            raise RuntimeError(self.error_message)

        if self.response_delay_sec > 0:
            time.sleep(self.response_delay_sec)

        # Extract evidence IDs from context
        evidence_ids = self._extract_evidence_ids(context)

        # Build answer based on query domain
        answer_dict = self._synthesize_investigation_payload(prompt, evidence_ids)

        if self.simulate_malformed_json:
            raw_str = json.dumps(answer_dict)
            return raw_str[: len(raw_str) // 2] + " ... [TRUNCATED_JSON"

        if self.simulate_plain_text:
            return (
                f"Investigation Analysis for query: {prompt}\n\n"
                f"Based on forensic evidence {', '.join(evidence_ids[:3])}, "
                f"the suspicious activity is confirmed. ATT&CK T1059.001 was identified. "
                "Phase 4 verification is required."
            )

        return json.dumps(answer_dict, indent=2)

    def generate_response(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate complete LLMResponse with token counts and parsed structured output."""
        start_t = time.perf_counter()
        try:
            raw_text = self.generate(
                prompt=prompt,
                context=context,
                system_prompt=system_prompt,
                **kwargs,
            )
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

            # Calculate deterministic pseudo-token counts based on character lengths
            context_len = len(str(context or ""))
            prompt_tokens = max(10, (len(prompt) + context_len) // 4)
            completion_tokens = max(10, len(raw_text) // 4)
            total_tokens = prompt_tokens + completion_tokens

            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            structured = InvestigationOutputParser.parse(
                raw_text=raw_text,
                model_name=self.model_name,
                token_usage=token_usage,
            )

            return LLMResponse(
                text=raw_text,
                structured=structured,
                model=self.model_name,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=duration_ms,
                error=None,
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
            logger.warning("MockLLM generation failed: %s", exc)
            return LLMResponse(
                text="",
                structured=None,
                model=self.model_name,
                provider=self.provider_name,
                latency_ms=duration_ms,
                error=str(exc),
            )

    def _extract_evidence_ids(self, context: list[dict[str, Any]] | str | None) -> list[str]:
        """Extract all evidence identifiers found within context."""
        ids: list[str] = []
        if isinstance(context, str):
            # Regex for [EVIDENCE-ID]
            found = re.findall(r"\[([A-Za-z0-9_\-\.\:\/]+)\]", context)
            for item in found:
                if any(k in item.upper() for k in ["EVID", "EVTX", "IMG", "DOCX", "PDF", "MORD", "FLOW", "ATTACK", "CVE", "T1", "UPLOAD"]):
                    ids.append(item)
        elif isinstance(context, list):
            for c in context:
                if isinstance(c, dict):
                    meta = c.get("metadata", {})
                    e_id = meta.get("evidence_id") or c.get("evidence_id")
                    if e_id:
                        ids.append(str(e_id))

        if not ids:
            ids = ["EVIDENCE-001", "EVIDENCE-002", "EVIDENCE-003"]

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        return deduped

    def _synthesize_investigation_payload(
        self,
        prompt: str,
        evidence_ids: list[str],
    ) -> dict[str, Any]:
        """Synthesize domain-appropriate response grounded in the supplied evidence IDs."""
        q_lower = prompt.lower()
        upload_ids = [i for i in evidence_ids if "UPLOAD" in i.upper()]
        kb_ids = [i for i in evidence_ids if "UPLOAD" not in i.upper()]

        # When both user upload and persistent KB are present, tailor synthesis to the user's specific request
        if upload_ids and kb_ids:
            up_id = upload_ids[0]
            kb_id1 = kb_ids[0]
            kb_id2 = kb_ids[1] if len(kb_ids) > 1 else kb_id1

            is_remediation = any(w in q_lower for w in ["fix", "remediate", "mitigate", "what should i do", "patch", "contain"])
            is_content_expl = any(w in q_lower for w in ["what this says", "what does this say", "explain this", "summarize", "what is this", "read this"])
            is_greeting = any(w in q_lower for w in ["hi", "hii", "hello", "hey"]) and len(q_lower.split()) <= 2

            if is_remediation:
                answer = (
                    f"Based on the threat telemetry observed in user-uploaded evidence [{up_id}] (USER_UPLOAD) and "
                    f"persistent cybersecurity knowledge-base advisories [{kb_id1}] (KNOWLEDGE_BASE), here are the recommended remediation actions:\n\n"
                    f"1. **Endpoint Containment**: Immediately isolate the host associated with [{up_id}] to prevent lateral adversary movement.\n"
                    f"2. **Process Termination & Persistence Removal**: Terminate suspicious processes and inspect scheduled tasks, services, and Run keys for persistence.\n"
                    f"3. **Credential Invalidation**: Reset passwords and revoke active Kerberos/OAuth session tokens for accounts observed in [{up_id}].\n"
                    f"4. **Network Perimeter Blocking**: Block all external C2 destination IP addresses and domains identified in the telemetry.\n"
                    f"5. **Hardening & Patching**: Apply vendor security patches as advised in [{kb_id2}] and enable PowerShell Constrained Language Mode and Script Block Logging."
                )
                claims = [
                    {
                        "claim_id": "CLAIM-001",
                        "text": f"Remediation actions target the active threat indicators observed in [{up_id}].",
                        "supporting_evidence_ids": [up_id],
                        "claim_type": "factual",
                    },
                    {
                        "claim_id": "CLAIM-002",
                        "text": f"Containment and mitigation protocols follow security advisories documented in [{kb_id1}].",
                        "supporting_evidence_ids": [kb_id1],
                        "claim_type": "inferential",
                    },
                    {
                        "claim_id": "CLAIM-003",
                        "text": f"Patching and credential hygiene measures address vulnerability exposure documented in [{kb_id2}].",
                        "supporting_evidence_ids": [kb_id2],
                        "claim_type": "factual",
                    },
                ]
            elif is_content_expl:
                answer = (
                    f"**Analysis of Uploaded Forensic Artifact ([{up_id}]):**\n\n"
                    f"The user-uploaded artifact [{up_id}] (USER_UPLOAD) documents observable security telemetry, command executions, "
                    f"and system events. Cross-referencing these indicators with persistent knowledge-base record [{kb_id1}] (KNOWLEDGE_BASE) "
                    f"identifies specific technique signatures matching documented threat activity. "
                    f"Advisory record [{kb_id2}] provides corresponding contextual background on the observed behaviors."
                )
                claims = [
                    {
                        "claim_id": "CLAIM-001",
                        "text": f"Artifact [{up_id}] contains observable security telemetry and command line activity.",
                        "supporting_evidence_ids": [up_id],
                        "claim_type": "factual",
                    },
                    {
                        "claim_id": "CLAIM-002",
                        "text": f"Observed activity in [{up_id}] matches technique signatures documented in persistent record [{kb_id1}].",
                        "supporting_evidence_ids": [up_id, kb_id1],
                        "claim_type": "inferential",
                    },
                    {
                        "claim_id": "CLAIM-003",
                        "text": f"Knowledge-base advisory [{kb_id2}] confirms threat context for the observed artifact.",
                        "supporting_evidence_ids": [kb_id2],
                        "claim_type": "factual",
                    },
                ]
            elif is_greeting:
                answer = (
                    f"Hello! I have loaded and analyzed your uploaded evidence [{up_id}] (USER_UPLOAD) alongside our persistent "
                    f"cybersecurity knowledge base ([{kb_id1}], [{kb_id2}]).\n\n"
                    f"**Artifact Summary:**\n"
                    f"The uploaded file records observable forensic indicators and system telemetry that cross-correlate with threat intelligence in [{kb_id1}].\n\n"
                    f"How would you like to proceed? I can:\n"
                    f"- 🔍 Detail what specific commands or events are documented in [{up_id}]\n"
                    f"- 🛡️ Explain the associated MITRE ATT&CK technique\n"
                    f"- 🔧 Recommend step-by-step remediation procedures"
                )
                claims = [
                    {
                        "claim_id": "CLAIM-001",
                        "text": f"User-uploaded evidence [{up_id}] was ingested and analyzed in the investigation context.",
                        "supporting_evidence_ids": [up_id],
                        "claim_type": "factual",
                    },
                    {
                        "claim_id": "CLAIM-002",
                        "text": f"Observed activity in [{up_id}] correlates with persistent knowledge-base record [{kb_id1}].",
                        "supporting_evidence_ids": [up_id, kb_id1],
                        "claim_type": "inferential",
                    },
                ]
            else:
                answer = (
                    f"Unified investigation synthesizing user-uploaded evidence [{up_id}] (USER_UPLOAD) with persistent "
                    f"knowledge-base threat intelligence [{kb_id1}] and [{kb_id2}] (KNOWLEDGE_BASE). "
                    f"The user-uploaded forensic artifact in [{up_id}] provides observed telemetry that directly correlates "
                    f"with adversarial techniques and system patterns documented in persistent knowledge base record [{kb_id1}]."
                )
                claims = [
                    {
                        "claim_id": "CLAIM-001",
                        "text": f"User-uploaded evidence in [{up_id}] documents observable forensic telemetry and commands.",
                        "supporting_evidence_ids": [up_id],
                        "claim_type": "factual",
                    },
                    {
                        "claim_id": "CLAIM-002",
                        "text": f"Observed activity in [{up_id}] cross-correlates with persistent knowledge-base technique [{kb_id1}].",
                        "supporting_evidence_ids": [up_id, kb_id1],
                        "claim_type": "inferential",
                    },
                    {
                        "claim_id": "CLAIM-003",
                        "text": f"Persistent cybersecurity knowledge-base record [{kb_id2}] confirms threat context and advisory guidance.",
                        "supporting_evidence_ids": [kb_id2],
                        "claim_type": "factual",
                    },
                ]

            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1003"] if any(k in q_lower for k in ["dump", "lsass", "mimikatz"]) else ["T1059.001"],
                "cves": ["CVE-2021-44228"] if any(k in q_lower for k in ["cve", "log4"]) else [],
                "confidence": "preliminary",
                "limitations": [
                    "User-uploaded evidence was evaluated jointly with persistent knowledge-base telemetry in a single context.",
                    "Phase 4 claim-level evidence verification audited claims against both provenance sources.",
                ],
            }

        top_ids = evidence_ids[:4]
        id1 = top_ids[0] if len(top_ids) > 0 else "EVIDENCE-001"
        id2 = top_ids[1] if len(top_ids) > 1 else id1
        id3 = top_ids[2] if len(top_ids) > 2 else id1

        # Check for SQL injection / SQLi question
        if "sql injection" in q_lower or ("sql" in q_lower and "injection" in q_lower) or "sqli" in q_lower:
            answer = (
                f"**SQL Injection (SQLi)** is one of the most critical web application vulnerabilities, "
                f"classified as **CWE-89** (OWASP A03:2021 – Injection).\n\n"
                f"**How it works:** Attackers insert malicious SQL statements into application input fields that are "
                f"directly concatenated into database queries without proper sanitization. This allows them to "
                f"bypass authentication, extract sensitive data, modify or delete database records, and in some "
                f"cases execute operating system commands via SQL stored procedures (ATT&CK T1505.001).\n\n"
                f"**Real-world example:** [{id1}] documents CVE-2023-34362, a critical SQLi vulnerability in "
                f"Progress MOVEit Transfer (CVSS 9.8) that was actively exploited by the Cl0p ransomware group.\n\n"
                f"**Attack variants:**\n"
                f"- In-Band SQLi (Error-based, UNION-based)\n"
                f"- Blind SQLi (Boolean-based, Time-based)\n"
                f"- Out-of-Band SQLi (DNS/HTTP exfiltration)\n\n"
                f"**Remediation:**\n"
                f"1. Use parameterized queries / prepared statements ([{id2}])\n"
                f"2. Implement strict input validation with allow-lists\n"
                f"3. Apply least-privilege database accounts\n"
                f"4. Deploy Web Application Firewalls (WAF) to detect automated SQLi tools"
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "SQL Injection is classified as CWE-89 and mapped to OWASP A03:2021 Injection category.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "CVE-2023-34362 (MOVEit Transfer SQLi) demonstrates critical real-world exploitation of SQL injection with CVSS 9.8.",
                    "supporting_evidence_ids": [id1, id2],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-003",
                    "text": "Parameterized queries and input validation are primary defenses against SQL injection attacks.",
                    "supporting_evidence_ids": [id2, id3],
                    "claim_type": "factual",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1190", "T1505.001"],
                "cves": ["CVE-2023-34362"],
                "confidence": "high",
                "limitations": [
                    "General cybersecurity knowledge answer grounded in retrieved vulnerability and ATT&CK records.",
                ],
            }

        # Check for Malware question
        if "malware" in q_lower:
            answer = (
                f"**Malware (Malicious Software)** is an overarching term for code or programs intentionally designed to "
                f"infiltrate systems, disrupt operations, steal sensitive data, or hijack computing resources ([{id1}]).\n\n"
                f"**Core Classifications:**\n"
                f"- **Trojans & Loaders**: Malicious payloads disguised as legitimate software ([{id2}])\n"
                f"- **Ransomware**: Extortion software encrypting files for ransom\n"
                f"- **Rootkits & Bootkits**: Stealth tools concealing unauthorized kernel/system access\n"
                f"- **Worms & Viruses**: Self-propagating and host-infecting code\n"
                f"- **Spyware & Keyloggers**: Covert activity surveillance and credential theft\n\n"
                f"**Infection Lifecycle:** Delivery (Phishing/Exploits) -> Execution ([{id2}]) -> Persistence -> Defense Evasion -> C2 Communication -> Actions on Objectives.\n\n"
                f"**Defenses & Countermeasures:**\n"
                f"1. Deploy Endpoint Detection & Response (EDR) with behavioral anomaly detection ([{id3}])\n"
                f"2. Enforce Application Whitelisting and software restriction policies\n"
                f"3. Disable legacy macros, script interpreters, and enforce PowerShell Constrained Language Mode\n"
                f"4. Apply network segmentation and outbound egress filtering"
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "Malware is malicious software designed to infiltrate systems, disrupt operations, or steal sensitive data.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Functional categories include trojans, ransomware, worms, viruses, rootkits, and spyware.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-003",
                    "text": "Endpoint Detection and Response (EDR) and application whitelisting are primary defenses against malware.",
                    "supporting_evidence_ids": [id1, id3],
                    "claim_type": "factual",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1588.001", "T1204", "T1059"],
                "cves": [],
                "confidence": "high",
                "limitations": [
                    "General cybersecurity reference answer grounded in authoritative concept definitions and MITRE ATT&CK records.",
                ],
            }

        # Check for Ransomware question
        if "ransomware" in q_lower:
            answer = (
                f"**Ransomware** is a destructive class of cyber extortion malware that systematically encrypts a victim organization's "
                f"files, databases, and network storage using strong cryptography (e.g. AES-256/RSA), demanding an extortion fee for decryption keys ([{id1}]).\n\n"
                f"**Extortion Models:**\n"
                f"- **Single Extortion**: Encryption of critical business data and systems\n"
                f"- **Double Extortion**: Pre-encryption data exfiltration with threats of public leak on dark web leak sites\n"
                f"- **Triple Extortion**: Adding DDoS attacks or directly contacting affected customers and regulators\n\n"
                f"**Adversary Playbook:** Initial access (phishing, exposed RDP, public vulnerabilities like MOVEit [{id3}]) -> Domain discovery -> Defense sabotage (`vssadmin delete shadows` / ATT&CK [{id2}]) -> Multi-threaded encryption -> Ransom note deployment.\n\n"
                f"**Remediation & Defense:**\n"
                f"1. Maintain offline, immutable 3-2-1 backups tested for rapid restoration\n"
                f"2. Implement rapid host isolation triggers via EDR upon canary file tampering\n"
                f"3. Disable internet-exposed RDP interfaces and enforce phishing-resistant MFA\n"
                f"4. Practice network micro-segmentation to restrict lateral spread"
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "Ransomware is extortion malware that systematically encrypts files and systems to demand a ransom.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Modern campaigns employ double extortion involving data exfiltration before encryption.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-003",
                    "text": "Air-gapped immutable backups and rapid network isolation are critical containment controls.",
                    "supporting_evidence_ids": [id1, id3],
                    "claim_type": "factual",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1486", "T1490", "T1190"],
                "cves": ["CVE-2023-34362"],
                "confidence": "high",
                "limitations": [
                    "General cybersecurity reference answer grounded in authoritative concept definitions and CISA advisories.",
                ],
            }

        # Check for Privilege Escalation question
        if "privilege escalation" in q_lower or ("privilege" in q_lower and "escalat" in q_lower):
            answer = (
                f"**Privilege Escalation** is the phase in an attack lifecycle where an adversary leverages vulnerabilities, configuration flaws, "
                f"or credential material to acquire higher permissions than originally granted ([{id1}]).\n\n"
                f"**Core Variants:**\n"
                f"- **Vertical Escalation**: Elevating from a standard user account to full administrative control (e.g. Windows `SYSTEM` or Linux `root`)\n"
                f"- **Horizontal Escalation**: Accessing accounts or resources belonging to peer users at the same privilege tier\n\n"
                f"**Common Attack Techniques:**\n"
                f"1. **Token Impersonation**: Abusing `SeImpersonatePrivilege` or `SeDebugPrivilege` (e.g. Potato exploits) ([{id2}])\n"
                f"2. **OS Kernel Exploits**: Triggering memory corruption in kernel drivers (e.g. Dirty COW, PrintNightmare)\n"
                f"3. **Misconfiguration Abuse**: Exploiting unquoted service paths, weak file ACLs, and sudoers wildcards\n"
                f"4. **UAC Bypass**: Abusing auto-elevating system binaries\n\n"
                f"**Defenses & Hardening:**\n"
                f"1. Strictly enforce the Principle of Least Privilege (PoLP)\n"
                f"2. Enable Windows Credential Guard and LSA Protection (RunAsPPL) ([{id3}])\n"
                f"3. Enforce rapid kernel patch management and driver blocklists\n"
                f"4. Regularly audit service configurations and file permissions"
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "Privilege escalation is the acquisition of elevated permissions via vulnerabilities, configuration flaws, or credential material.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Common mechanisms include token impersonation, kernel exploitation, service misconfigurations, and UAC bypass.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-003",
                    "text": "Enforcing the principle of least privilege and LSA protection are effective controls against unauthorized elevation.",
                    "supporting_evidence_ids": [id1, id3],
                    "claim_type": "factual",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1548", "T1068", "T1134"],
                "cves": [],
                "confidence": "high",
                "limitations": [
                    "General cybersecurity reference answer grounded in authoritative concept definitions and MITRE ATT&CK records.",
                ],
            }

        # Check for phishing question
        if "phishing" in q_lower:
            answer = (
                f"**Phishing** is a deceptive social engineering attack where adversaries disguise themselves as trusted "
                f"entities to manipulate targets into revealing sensitive credentials, financial details, or executing malicious payloads ([{id1}]).\n\n"
                f"**Attack Vectors & Variants:**\n"
                f"- **Spearphishing**: Tailored targeting of specific high-value individuals ([{id2}])\n"
                f"- **Whaling**: Targeting senior executives (C-level) and financial controllers\n"
                f"- **Business Email Compromise (BEC)**: Fraudulent payment and invoice redirection\n"
                f"- **Smishing & Vishing**: SMS and voice-based social engineering\n\n"
                f"**Technical Indicators:** Lookalike domains, forged headers, credential harvesting portals, and malicious macro/HTML attachments ([{id3}]).\n\n"
                f"**Defenses & Countermeasures:**\n"
                f"1. Enforce phishing-resistant Multi-Factor Authentication (FIDO2 / WebAuthn)\n"
                f"2. Strict email authentication protocols: SPF, DKIM, and DMARC reject policies\n"
                f"3. Secure Email Gateways (SEG) with real-time URL sandboxing\n"
                f"4. Continuous security awareness training and phishing simulations"
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "Phishing is a social engineering attack mechanism categorized under ATT&CK Initial Access (T1566) and Reconnaissance (T1598).",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Common attack vectors include spearphishing emails, credential harvesting links, and malicious attachments.",
                    "supporting_evidence_ids": [id1, id2],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-003",
                    "text": "Phishing-resistant MFA (FIDO2) and DMARC email authentication are primary technical defenses.",
                    "supporting_evidence_ids": [id2, id3],
                    "claim_type": "factual",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1566", "T1598"],
                "cves": [],
                "confidence": "high",
                "limitations": [
                    "General cybersecurity reference answer grounded in authoritative concept definitions and MITRE ATT&CK records.",
                ],
            }

        # Check for standalone remediation question
        if any(w in q_lower for w in ["fix", "remediate", "mitigate", "what should i do"]):
            answer = (
                f"Based on retrieved cybersecurity threat intelligence [{id1}] and advisory guidance [{id2}], "
                f"the standard containment and remediation procedures include:\n\n"
                f"1. **Isolate Affected Systems**: Disconnect compromised hosts from the network immediately.\n"
                f"2. **Revoke & Reset Credentials**: Invalidate active sessions and reset passwords for compromised accounts.\n"
                f"3. **Kill Malicious Processes**: Terminate suspect processes and purge persistence artifacts.\n"
                f"4. **Block C2 Indicators**: Add malicious IPs/domains to perimeter firewall and DNS filtering rules.\n"
                f"5. **Patch Vulnerabilities**: Apply vendor security updates documented in [{id3}] and enforce least privilege."
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "Standard incident response protocol requires immediate endpoint isolation and credential invalidation.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Perimeter IOC blocking and system hardening prevent reinfection and lateral movement.",
                    "supporting_evidence_ids": [id2, id3],
                    "claim_type": "inferential",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": [],
                "cves": [],
                "confidence": "high",
                "limitations": [
                    "Remediation guidance synthesized from established cybersecurity frameworks and advisories.",
                ],
            }

        if "cve-2021-44228" in q_lower or "log4j" in q_lower or "log4shell" in q_lower:
            answer = (
                f"Retrieved technical evidence from [{id1}] establishes CVE-2021-44228 (Log4Shell) as a critical "
                "remote code execution vulnerability in Apache Log4j 2.x JNDI lookup functionality. "
                f"Corroborating security telemetry in [{id2}] indicates exploit attempts delivering malicious payloads. "
                f"Forensic advisory in [{id3}] notes CVSS base score 10.0 with immediate patch advisories."
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "CVE-2021-44228 is an unauthenticated Remote Code Execution flaw in Apache Log4j JNDI lookups.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Exploitation telemetry indicates malicious outbound LDAP/RMI directory requests.",
                    "supporting_evidence_ids": [id2, id3],
                    "claim_type": "inferential",
                },
            ]
            attck = ["T1190"]
            cves = ["CVE-2021-44228"]

        elif "credential" in q_lower or "mimikatz" in q_lower or "dump" in q_lower or "lsass" in q_lower:
            answer = (
                f"Retrieved forensic evidence in [{id1}] and [{id2}] provides direct evidence of credential dumping activity. "
                "Process telemetry records suspicious memory access targeting the Local Security Authority Subsystem Service (LSASS). "
                f"Security advisories and telemetry in [{id1}] document OS credential dumping techniques (T1003)."
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "LSASS memory read access was detected, consistent with credential harvesting.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Adversary credential dumping techniques targeting OS memory structures (T1003) were identified.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "attribution",
                },
            ]
            attck = ["T1003"]
            cves = []

        elif ("log" in q_lower and "log4" not in q_lower) or "telemetry" in q_lower or "event log" in q_lower:
            answer = (
                f"**Security Event Logs** are timestamped digital records generated by operating systems, applications, "
                f"network devices, and security controls that document state changes, transactions, and user activities ([{id1}]).\n\n"
                f"**Key Categories in Cybersecurity:**\n"
                f"- **Host & OS Logs**: Windows Event Logs (EVTX) and Sysmon recording process creations (Event ID 1), network connections (Event ID 3), and logon events ([{id2}])\n"
                f"- **Authentication & Audit Trails**: Linux `/var/log/auth.log`, `auditd`, and domain controller directory access\n"
                f"- **Network Telemetry**: Firewall connection logs, DNS queries, and NetFlow traffic summaries\n"
                f"- **Application & Web Logs**: HTTP access logs capturing web requests and injection attempts\n\n"
                f"**Role in Investigations:** Logs provide empirical ground truth for timeline reconstruction, root-cause analysis, lateral movement tracking, and compliance audits ([{id3}]).\n\n"
                f"**Best Practices:** Centralize logs via a SIEM/data lake, enforce append-only remote shipping to prevent adversary tampering (ATT&CK T1070), and synchronize clocks via NTP."
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "Security event logs are timestamped records documenting operating system, network, and application activities.",
                    "supporting_evidence_ids": [id1],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Windows Sysmon and Linux audit logs capture critical process creation and authentication telemetry.",
                    "supporting_evidence_ids": [id1, id2],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-003",
                    "text": "Centralized logging and append-only shipping prevent adversary log tampering and deletion.",
                    "supporting_evidence_ids": [id1, id3],
                    "claim_type": "factual",
                },
            ]
            return {
                "answer": answer,
                "claims": claims,
                "attck_techniques": ["T1070", "T1562.002"],
                "cves": [],
                "confidence": "high",
                "limitations": [
                    "General cybersecurity reference answer grounded in authoritative logging standards and NIST guidelines.",
                ],
            }

        elif "powershell" in q_lower or "bypass" in q_lower:
            answer = (
                f"Retrieved endpoint telemetry in [{id1}] indicates PowerShell execution with policy bypass arguments. "
                f"Security event logs in [{id2}] capture script block logging activity and encoded command execution. "
                f"Visual SIEM alert capture in [{id3}] displays Critical severity notification for ExecutionPolicy Bypass."
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": "PowerShell was spawned with execution policy bypass parameters.",
                    "supporting_evidence_ids": [id1, id3],
                    "claim_type": "factual",
                },
                {
                    "claim_id": "CLAIM-002",
                    "text": "Script block telemetry indicates automated reconnaissance and discovery script invocation.",
                    "supporting_evidence_ids": [id2],
                    "claim_type": "inferential",
                },
            ]
            attck = ["T1059.001"]
            cves = []

        else:
            answer = (
                f"Based on the retrieved cybersecurity reference knowledge [{id1}] and corroborating evidence [{id2}], "
                f"here is the analysis for '{prompt}':\n\n"
                f"The evidence base documents security mechanisms and operational procedures associated with this topic. "
                f"Cross-referencing authoritative guidance in [{id3}] provides foundational context and detection telemetry."
            )
            claims = [
                {
                    "claim_id": "CLAIM-001",
                    "text": f"Retrieved cybersecurity records provide foundational context for '{prompt}'.",
                    "supporting_evidence_ids": [id1, id2],
                    "claim_type": "factual",
                }
            ]
            attck = []
            cves = []

        return {
            "answer": answer,
            "claims": claims,
            "attck_techniques": attck,
            "cves": cves,
            "confidence": "high",
            "limitations": [
                "Analysis grounded in retrieved canonical cybersecurity evidence and threat framework records.",
            ],
        }
