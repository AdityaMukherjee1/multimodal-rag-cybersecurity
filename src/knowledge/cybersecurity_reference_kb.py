"""Authoritative Cybersecurity Reference & Explanatory Knowledge Base.

Provides structured, canonical explanatory records for foundational cybersecurity
domains, threat mechanisms, and concepts (e.g., Malware, Ransomware, Phishing,
Privilege Escalation, DoS/DDoS, Credential Dumping, Zero-Day Vulnerabilities)
alongside core software weaknesses (CWE/OWASP).

Guarantees that general educational queries retrieve authoritative conceptual definitions
and reference explanations alongside supporting MITRE ATT&CK techniques and CVE records,
with strict epistemic categorization (REFERENCE_KNOWLEDGE, never OBSERVED_EVIDENCE).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from src.ingestion.evidence_schema import CanonicalEvidenceRecord
from src.knowledge.vulnerability_kb import VulnerabilityKnowledgeBase
from src.retrieval.vector_store import RetrievalResult

logger = logging.getLogger(__name__)

# Foundational cybersecurity reference definitions aligned with NIST SP 800-61r2, CISA, and MITRE ATT&CK
_CORE_REFERENCE_TOPICS: list[dict[str, Any]] = [
    {
        "topic_id": "REF-PHISHING",
        "name": "Phishing & Social Engineering",
        "category": "Initial Access / Human Manipulation",
        "related_attck": ["T1566", "T1598"],
        "description": (
            "Phishing is a deceptive social engineering attack mechanism where adversaries disguise themselves as "
            "trusted entities (such as organizations, colleagues, or service providers) to manipulate human targets "
            "into divulging sensitive credentials, financial details, or executing malicious attachments and links. "
            "Attack vectors encompass Spearphishing (tailored targeting of specific high-value individuals), Whaling "
            "(targeting senior executives), Smishing (SMS phishing), Vishing (voice deception), and Business Email "
            "Compromise (BEC) involving fraudulent payment requests."
        ),
        "technical_mechanics": [
            "Domain Spoofing & Typosquatting: Registering lookalike domains with deceptive character substitutions.",
            "Deceptive Pretexting: Fabricating urgent scenarios (e.g., password reset, unpaid invoice, HR notice).",
            "Credential Harvesting Clones: Deploying reverse proxies or replica login portals to capture passwords and MFA session cookies.",
            "Payload Delivery: Attaching weaponized PDF/Office documents with malicious macros, exploit code, or HTML smuggling.",
        ],
        "defenses_and_countermeasures": [
            "Phishing-resistant Multi-Factor Authentication (MFA) utilizing FIDO2 / WebAuthn hardware keys.",
            "Email authentication protocol enforcement: Sender Policy Framework (SPF), DomainKeys Identified Mail (DKIM), and DMARC reject policies.",
            "Automated Email Security Gateways (SEG) with real-time URL detonation and attachment sandboxing.",
            "Continuous security awareness simulations and user reporting channels.",
        ],
        "real_world_examples": [
            "FIDO2-bypassing adversary-in-the-middle (AiTM) reverse proxy phishing campaigns.",
            "CISA Alert AA23-158A (Phishing vectors delivering Cl0p ransomware and initial access trojans).",
        ],
    },
    {
        "topic_id": "REF-MALWARE",
        "name": "Malware (Malicious Software)",
        "category": "Malicious Code / Threat Vectors",
        "related_attck": ["T1588.001", "T1204", "T1059"],
        "description": (
            "Malware (malicious software) is an overarching term for any software, script, or firmware intentionally "
            "designed to infiltrate computer systems, disrupt operations, exfiltrate sensitive data, gain unauthorized "
            "access, or hijack computational resources without the owner's informed consent. Core functional classifications "
            "include Viruses (self-replicating host inserters), Worms (network-propagating standalone programs), Trojans "
            "(malicious payloads masquerading as benign utilities), Ransomware (extortion cryptors), Spyware and Keyloggers "
            "(covert surveillance), Rootkits (kernel and bootloader stealth tools), and Backdoors/C2 Implants."
        ),
        "technical_mechanics": [
            "Infection Lifecycle: Initial delivery -> Execution -> Persistence (scheduled tasks, services) -> Defense Evasion (packing, obfuscation, API unhooking) -> Command & Control (C2 beaconing) -> Actions on Objectives.",
            "Evasion Techniques: In-memory reflective DLL loading, process hollowing, AMSI/ETW patching, and sleep obfuscation.",
            "Payload Staging: Droppers downloading secondary stages from encrypted C2 channels.",
        ],
        "defenses_and_countermeasures": [
            "Endpoint Detection and Response (EDR) utilizing behavioral heuristics and kernel callbacks.",
            "Application Whitelisting / AppLocker / Software Restriction Policies.",
            "Disabling legacy macros, script interpreters (VBScript, HTA), and enforcing PowerShell Constrained Language Mode.",
            "Network segmentation, micro-segmentation, and egress filtering to sever C2 communications.",
        ],
        "real_world_examples": [
            "Emotet, TrickBot, and QakBot banking trojans and modular malware loaders.",
            "Stuxnet industrial control system rootkit and worm.",
        ],
    },
    {
        "topic_id": "REF-RANSOMWARE",
        "name": "Ransomware & Cyber Extortion",
        "category": "Extortion Malware / Impact",
        "related_attck": ["T1486", "T1490", "T1190"],
        "description": (
            "Ransomware is a severe category of extortion malware that systematically discovers, accesses, and encrypts a "
            "victim organization's critical files, databases, virtual machine disks, and shared network storage using military-grade "
            "cryptography (such as AES-256 and RSA-4096), demanding an extortion fee in exchange for the decryption keys. Modern "
            "ransomware operations deploy multi-tiered extortion strategies: Double Extortion (exfiltrating confidential proprietary "
            "data prior to encryption and threatening public disclosure on dark web leak sites), and Triple Extortion (combining "
            "encryption, data leaks, and aggressive DDoS attacks against victim customer services)."
        ),
        "technical_mechanics": [
            "Pre-Encryption Reconnaissance: Domain enumeration, Active Directory discovery (BloodHound, AdFind), and backup location identification.",
            "Defense & Recovery Sabotage: Purging Windows Volume Shadow Copies ('vssadmin delete shadows'), disabling system recovery, and tampering with backup agents.",
            "High-Speed Asymmetric Encryption: Utilizing multi-threaded cryptographic APIs to encrypt file headers and append distinctive extensions.",
            "Automated Lateral Movement: Scripted propagation via PsExec, WMI, Group Policy Objects (GPO), or compromised administrative RDP credentials.",
        ],
        "defenses_and_countermeasures": [
            "Air-gapped, immutable, offline backups adhering strictly to the 3-2-1 backup principle.",
            "Rapid network isolation playbooks and automated containment triggers via EDR.",
            "Disabling exposed RDP interfaces and enforcing MFA across all remote access gateways.",
            "Tiered administrative access models and Privileged Access Management (PAM).",
        ],
        "real_world_examples": [
            "CL0P Ransomware Group (exploiting CVE-2023-34362 MOVEit Transfer - CISA AA23-158A).",
            "LockBit, BlackCat (ALPHV), WannaCry, and DarkSide ransomware strains.",
        ],
    },
    {
        "topic_id": "REF-PRIV_ESCALATION",
        "name": "Privilege Escalation",
        "category": "Privilege Escalation / Exploitation",
        "related_attck": ["T1548", "T1068", "T1134", "T1546"],
        "description": (
            "Privilege Escalation is the deliberate phase in an attack lifecycle where an adversary leverages software "
            "vulnerabilities, operating system design flaws, misconfigurations, or credential material to obtain higher "
            "clearance or permissions than originally granted. It is categorized into Vertical Escalation (transitioning "
            "from a standard unprivileged user account to full administrative or root/SYSTEM authority) and Horizontal "
            "Escalation (accessing resources or accounts belonging to peer users at the same privilege tier)."
        ),
        "technical_mechanics": [
            "Token Manipulation & Impersonation: Abusing elevated Windows privileges such as SeDebugPrivilege or SeImpersonatePrivilege (e.g., RottenPotato, PrintSpoofer).",
            "Operating System Kernel Exploits: Triggering memory corruption, race conditions, or integer overflows in kernel drivers (e.g., CVE-2021-40449, Dirty COW).",
            "Configuration Abuses: Exploiting unquoted service executable paths, weak file permissions on service binaries, and writable PATH directories.",
            "UAC Bypass: Abusing auto-elevating Windows binaries and registry redirection to execute commands without consent prompts.",
            "Linux Sudo Misconfigurations: Exploiting wildcards in /etc/sudoers or SUID root binaries.",
        ],
        "defenses_and_countermeasures": [
            "Principle of Least Privilege (PoLP): Stripping administrative rights from standard user accounts.",
            "Credential hygiene: Enforcing Windows LSA Protection (RunAsPPL) and Credential Guard.",
            "Prompt operating system kernel patch management and driver blocklist enforcement (HVCI).",
            "Auditing service configurations, file system access control lists (ACLs), and sudo permissions.",
        ],
        "real_world_examples": [
            "Abuse of SeImpersonatePrivilege via Potato exploit families to elevate from IIS service account to NT AUTHORITY\\SYSTEM.",
            "CVE-2021-1675 / CVE-2021-34527 (PrintNightmare Windows Print Spooler privilege escalation and RCE).",
        ],
    },
    {
        "topic_id": "REF-DDOS",
        "name": "Denial of Service (DoS & DDoS)",
        "category": "Availability / Impact",
        "related_attck": ["T1498", "T1499"],
        "description": (
            "A Denial of Service (DoS) or Distributed Denial of Service (DDoS) attack is a malicious attempt to disrupt the "
            "normal traffic of a targeted server, service, or network by overwhelming the target or its surrounding infrastructure "
            "with a flood of internet traffic. While a standard DoS originates from a single source, a DDoS orchestrates distributed "
            "botnets comprising thousands or millions of compromised IoT devices and servers globally to saturate bandwidth and compute capacity."
        ),
        "technical_mechanics": [
            "Volumetric Attacks: Saturating network bandwidth through high-volume UDP floods, ICMP echo floods, and DNS/NTP/SNMP reflection amplification.",
            "Protocol Attacks: Consuming state tables of firewalls, load balancers, and web servers via SYN floods, Ping of Death, and Smurf attacks.",
            "Application Layer (Layer 7) Attacks: Overwhelming application web servers with computationally intensive HTTP GET/POST floods or Slowloris connections.",
        ],
        "defenses_and_countermeasures": [
            "Deploying Cloud DDoS Mitigation and scrubbing networks (Cloudflare, AWS Shield, Akamai).",
            "Implementing Anycast DNS and content delivery networks (CDNs) to distribute traffic loads geographically.",
            "Configuring rate limiting, Web Application Firewall (WAF) challenge rules, and SYN cookies on boundary load balancers.",
        ],
        "real_world_examples": [
            "Mirai Botnet DDoS attacks leveraging default-credential IoT devices.",
            "Record-breaking multi-terabit HTTP/2 Rapid Reset DDoS attacks (CVE-2023-44487).",
        ],
    },
    {
        "topic_id": "REF-CREDENTIAL_DUMPING",
        "name": "Credential Dumping & Identity Theft",
        "category": "Credential Access",
        "related_attck": ["T1003", "T1003.001", "T1003.002", "T1003.006"],
        "description": (
            "Credential Dumping is the adversary technique of extracting authentication credentials (such as plaintext passwords, "
            "NTLM password hashes, Kerberos tickets, and DPAPI private keys) from system volatile memory, system storage, or Active "
            "Directory databases. Once dumped, credentials allow adversaries to authenticate as legitimate users, bypass controls, "
            "and execute lateral movement across the enterprise."
        ),
        "technical_mechanics": [
            "LSASS Memory Access: Opening handles to Local Security Authority Subsystem Service (lsass.exe) memory to read cached credentials (Mimikatz, MiniDumpWriteDump, ProcDump).",
            "Security Account Manager (SAM) Extraction: Dumping local Windows user hashes from the registry hive (HKLM\\SAM, HKLM\\SYSTEM).",
            "DCSync Attacks: Simulating domain controller replication traffic (MS-DRSR) to remotely harvest all domain user password hashes.",
            "Kerberoasting: Requesting Kerberos service tickets (TGS) for accounts with Service Principal Names (SPNs) and cracking ticket hashes offline.",
        ],
        "defenses_and_countermeasures": [
            "Enabling Windows Defender Credential Guard (virtualization-based isolation of LSASS).",
            "Enforcing LSA RunAsPPL (Protected Process Light) to block non-protected processes from reading LSASS memory.",
            "Disabling obsolete authentication protocols (NTLMv1, WDigest, LM) in favor of modern Kerberos.",
            "Monitoring Event ID 4662 (DCSync replication rights access) and Sysmon Event ID 10 (ProcessAccess targeting lsass.exe).",
        ],
        "real_world_examples": [
            "Mimikatz extraction of ClearText credentials and Golden Ticket generation in Active Directory compromises.",
        ],
    },
    {
        "topic_id": "REF-ZERO_DAY",
        "name": "Zero-Day Vulnerability & Exploit",
        "category": "Vulnerability / Exploit Lifecycle",
        "related_attck": ["T1190", "T1203"],
        "description": (
            "A Zero-Day vulnerability is an undisclosed computer-software security weakness that is unknown to the software vendor "
            "or developers, or known but for which no official vendor patch or security update is currently available. A Zero-Day Exploit "
            "is the working attack code that weaponizes this vulnerability. Because no signature or official remediation exists during "
            "the zero-day window, attacks possess an exceptionally high probability of success against unprotected targets."
        ),
        "technical_mechanics": [
            "Vulnerability Discovery: Adversary reverse engineering, fuzzing, or acquiring undisclosed bugs on broker markets.",
            "Exploit Development: Crafting memory corruptions (buffer overflows, use-after-free), logic flaws, or authentication bypasses that evade standard defenses.",
            "Window of Exposure: The timeframe between initial exploit release in the wild and the release and broad installation of the vendor patch.",
        ],
        "defenses_and_countermeasures": [
            "Defense-in-Depth and zero-trust architectural segmentation limiting blast radius.",
            "Behavioral anomaly detection and heuristic threat hunting rather than static signature matching.",
            "Virtual Patching: Deploying WAF and IPS inspection rules that detect and neutralize the exploit payload pattern prior to vendor patch availability.",
            "Endpoint exploit mitigation controls (ASLR, DEP, Control Flow Guard, CET).",
        ],
        "real_world_examples": [
            "Log4Shell (CVE-2021-44228) zero-day RCE in Apache Log4j JNDI lookup functionality.",
            "MOVEit Transfer SQL Injection zero-day (CVE-2023-34362) actively weaponized by extortion actors.",
        ],
    },
    {
        "topic_id": "REF-SECURITY_LOGS",
        "name": "Security Event Logs & Audit Telemetry",
        "category": "Telemetry & Observability / Digital Forensics",
        "related_attck": ["T1070", "T1562.002", "T1059"],
        "description": (
            "Security event logs are immutable, timestamped digital records generated by operating systems, applications, "
            "network appliances, and security sensors that document system state transitions, user activities, and transactional events. "
            "In cybersecurity incident investigation, logs provide the empirical ground truth required for timeline reconstruction, "
            "intrusion root-cause analysis, lateral movement tracking, adversary attribution, and compliance validation."
        ),
        "technical_mechanics": [
            "Operating System Telemetry: Windows Event Logs (EVTX) including Sysmon Event ID 1 (Process Creation), ID 3 (Network Connect), ID 7 (Image Load), and Security Event ID 4624 (Logon).",
            "Linux & UNIX Logging: systemd journal, /var/log/auth.log, syslog, and auditd system call monitoring.",
            "Network & Perimeter Telemetry: NetFlow/IPFIX, firewall connection logs, DNS query logs, and proxy access records.",
            "Centralized SIEM Aggregation: Ingesting, parsing, normalizing, and correlating heterogeneous log streams into structured security analytics.",
        ],
        "defenses_and_countermeasures": [
            "Centralized, write-once, append-only remote log shipping to prevent local adversary tampering or log clearing (T1070).",
            "Enforcing granular audit policies and deploying Microsoft Sysmon across endpoints.",
            "Configuring automated SIEM alerting rules for critical event IDs (e.g., Event ID 1102 log cleared, Event ID 4720 account created).",
            "Enforcing secure Network Time Protocol (NTP) synchronization across all infrastructure components.",
        ],
        "real_world_examples": [
            "Sysmon process execution telemetry identifying Meterpreter payload staging and encoded PowerShell execution.",
            "Firewall traffic logs revealing adversary Command and Control (C2) beaconing patterns.",
        ],
    },
    {
        "topic_id": "REF-FIREWALL",
        "name": "Firewalls & Network Boundary Defense",
        "category": "Network Security / Boundary Defense",
        "related_attck": ["T1562.004", "T1071"],
        "description": (
            "A firewall is a core network security system that monitors and controls incoming and outgoing network traffic "
            "based on predetermined security rules, establishing a barrier between a trusted internal network and untrusted external networks.",
        ),
        "technical_mechanics": [
            "Stateful Packet Inspection: Tracking active TCP/UDP connections to block unsolicited inbound packets.",
            "Next-Generation Firewalls (NGFW): Deep packet inspection, application-level visibility, and integrated IPS.",
            "Web Application Firewalls (WAF): Inspecting Layer 7 HTTP/HTTPS traffic to filter OWASP Top 10 exploits.",
        ],
        "defenses_and_countermeasures": [
            "Enforcing default-deny ingress and egress filtering policies.",
            "Regular firewall rule auditing and unused port closure.",
            "Deploying TLS decryption for deep traffic inspection.",
        ],
        "real_world_examples": [
            "Perimeter firewall dropping adversary reverse TCP connection attempts on unapproved ports.",
        ],
    },
]


class CybersecurityReferenceKnowledgeBase:
    """Manages authoritative cybersecurity reference concepts, definitions, and weakness classifications."""

    _cached_records: list[CanonicalEvidenceRecord] | None = None
    _cached_embeddings: np.ndarray | None = None
    _embedder: Any | None = None

    @classmethod
    def get_canonical_records(cls) -> list[CanonicalEvidenceRecord]:
        """Return all canonical reference and vulnerability records."""
        if cls._cached_records is not None:
            return cls._cached_records

        records: list[CanonicalEvidenceRecord] = []

        # 1. Ingest Core Conceptual Reference Topics
        for idx, item in enumerate(_CORE_REFERENCE_TOPICS):
            t_id = item["topic_id"]
            name = item["name"]
            cat = item["category"]
            attcks = item["related_attck"]
            desc = item["description"]
            mech = "\n".join(f"- {m}" for m in item["technical_mechanics"])
            defs = "\n".join(f"{i}. {d}" for i, d in enumerate(item["defenses_and_countermeasures"], 1))
            cves_ex = ", ".join(item.get("real_world_examples", []))
            attck_str = ", ".join(attcks)

            content = (
                f"[CYBERSECURITY REFERENCE KNOWLEDGE: CONCEPT & DEFINITION] {name}\n"
                f"Classification: {cat} | Primary MITRE ATT&CK Mapping: {attck_str}\n"
                f"Provenance: AUTHORITATIVE REFERENCE KNOWLEDGE (KNOWLEDGE_BASE)\n\n"
                f"Comprehensive Definition & Overview:\n{desc}\n\n"
                f"Technical Attack Mechanics & Vectors:\n{mech}\n\n"
                f"Recommended Defenses, Countermeasures & Best Practices:\n{defs}\n\n"
                f"Notable Threat Examples & Case Studies:\n{cves_ex}"
            )

            rec = CanonicalEvidenceRecord(
                evidence_id=t_id,
                source_file="cybersecurity_reference_handbook.json",
                source_dataset="reference_knowledge",
                modality="reference_handbook",
                content=content,
                raw_reference="data/knowledge/cybersecurity_reference_handbook.json",
                metadata={
                    "topic_id": t_id,
                    "topic_name": name,
                    "category": cat,
                    "attck_techniques": attcks,
                    "attck_technique": attcks[0] if attcks else None,
                    "provenance_type": "KNOWLEDGE_BASE",
                    "evidence_nature": "REFERENCE_KNOWLEDGE",
                    "is_reference_knowledge": True,
                },
                attck_technique=attcks[0] if attcks else None,
                event_type="reference_definition",
                chunk_id=idx,
            )
            records.append(rec)

        # 2. Ingest CWE / OWASP Vulnerability Knowledge Base Records
        vulnerability_records = VulnerabilityKnowledgeBase.get_canonical_records()
        for vrec in vulnerability_records:
            vrec.metadata["provenance_type"] = "KNOWLEDGE_BASE"
            vrec.metadata["evidence_nature"] = "REFERENCE_KNOWLEDGE"
            vrec.metadata["is_reference_knowledge"] = True
            vrec.source_dataset = "reference_knowledge"
            records.append(vrec)

        cls._cached_records = records
        logger.info("Initialized CybersecurityReferenceKnowledgeBase with %d canonical reference records.", len(records))
        return records

    @classmethod
    def search_reference_knowledge(
        cls,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 2,
        min_score: float = 0.55,
    ) -> list[RetrievalResult]:
        """Perform semantic and keyword search against the reference knowledge base.

        Returns top-matching canonical reference definitions formatted as RetrievalResult
        with evidence_nature='REFERENCE_KNOWLEDGE' and provenance_type='KNOWLEDGE_BASE'.
        """
        records = cls.get_canonical_records()
        q_lower = query.lower()

        # Stop words
        stop_words = {
            "what", "is", "a", "an", "the", "and", "or", "in", "on", "of", "to",
            "for", "with", "this", "that", "these", "those", "how", "explain",
            "describe", "does", "do", "can", "about", "tell", "me", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "it", "its",
            "my", "your", "we", "they", "you", "i", "define", "meaning",
        }
        q_tokens = [w for w in re.findall(r"\b\w+\b", q_lower) if w not in stop_words and len(w) > 1]

        # Lazy initialize embeddings if query embedding is provided
        sim_scores: np.ndarray | None = None
        if query_embedding is not None:
            if cls._cached_embeddings is None:
                try:
                    from src.embeddings.embedding_service import EmbeddingService
                    embedder = cls._embedder or EmbeddingService()
                    cls._embedder = embedder
                    texts = [r.content for r in records]
                    cls._cached_embeddings = embedder.embed_texts(texts)
                except Exception as exc:
                    logger.warning("Failed to embed reference knowledge: %s", exc)

            if cls._cached_embeddings is not None:
                try:
                    sim_scores = (cls._cached_embeddings @ query_embedding.T).flatten()
                except Exception as exc:
                    logger.warning("Dot product failed for reference search: %s", exc)

        results: list[tuple[float, CanonicalEvidenceRecord]] = []
        for idx, rec in enumerate(records):
            base_score = float(sim_scores[idx]) if sim_scores is not None else 0.50
            combined_text = (rec.content + " " + str(rec.metadata)).lower()

            # Keyword overlap ratio
            token_matches = sum(1 for tok in q_tokens if tok in combined_text)
            overlap_ratio = token_matches / len(q_tokens) if q_tokens else 0.0

            # Compute hybrid score
            hybrid_score = base_score * (1.0 + 0.35 * overlap_ratio)

            # Direct topic matching boosts
            topic_name = rec.metadata.get("topic_name", rec.metadata.get("vulnerability_name", "")).lower()
            if any(tok in topic_name for tok in q_tokens):
                hybrid_score = max(hybrid_score, 0.88 + 0.05 * overlap_ratio)

            # Specific high-relevance domain keyword matches
            if "phish" in q_lower and "phishing" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)
            elif "ransomware" in q_lower and "ransomware" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)
            elif "malware" in q_lower and "malware" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)
            elif "privilege escalation" in q_lower and "privilege escalation" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)
            elif "sql" in q_lower and "sql" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)
            elif ("ddos" in q_lower or "denial of service" in q_lower) and ("denial of service" in topic_name or "dos" in topic_name):
                hybrid_score = max(hybrid_score, 0.95)
            elif ("credential dumping" in q_lower or "lsass" in q_lower) and "credential" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)
            elif ("zero-day" in q_lower or "zero day" in q_lower) and "zero-day" in topic_name:
                hybrid_score = max(hybrid_score, 0.95)

            if hybrid_score >= min_score:
                results.append((hybrid_score, rec))

        results.sort(key=lambda x: x[0], reverse=True)

        retrieval_results: list[RetrievalResult] = []
        for rank, (score, rec) in enumerate(results[:top_k], start=1):
            doc = rec.to_document_dict()
            doc_meta = dict(doc["metadata"])
            doc_meta["provenance_type"] = "KNOWLEDGE_BASE"
            doc_meta["evidence_nature"] = "REFERENCE_KNOWLEDGE"
            doc_meta["source_dataset"] = "reference_knowledge"
            doc_meta["is_reference_knowledge"] = True

            retrieval_results.append(
                RetrievalResult(
                    content=doc["content"],
                    metadata=doc_meta,
                    score=min(score, 0.99),
                    rank=rank,
                )
            )

        return retrieval_results
