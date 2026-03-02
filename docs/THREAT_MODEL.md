# Kavach Threat Model

This document outlines the operational security boundaries of the Kavach Gateway. It serves to clarify to engineering and compliance teams exactly which threats the system natively mitigates, which it degrades gracefully against, and which remain inherently out-of-scope for the proxy layer.

---

## 🛡️ Threats Handled (In-Scope)

Kavach operates as a layered defense mechanism specifically focused on securing LLM input, output, and tool invocations.

1. **Prompt Injection & Jailbreaks**
   - **Vector:** Attackers attempt to override system instructions via 'ignore previous instructions' triggers, roleplay frameworks (DAN), or ethical bypasses.
   - **Mitigation:** Handled via deterministic AST rule parsing alongside a low-latency machine learning ensemble comprising gradient boosting, logistic regression, and continuous offline encoding embeddings.

2. **Advanced Persistent Obfuscation (APTs)**
   - **Vector:** Adversaries encode malicious prompts using nested chains (Base64 -> Hex), stealth Zero-Width Characters (ZWC Steganography), or Cyrillic/Greek homoglyph spoofing to evade regex scanners while still compiling into LLM sub-word tokens.
   - **Mitigation:** The `APTDetector` physically strips, remaps, and unrolls layered formatting prior to structural analysis.

3. **Data Exfiltration & Financial PII Leaks**
   - **Vector:** An LLM generates sensitive context data or is instructed to send local state to external unverified endpoints.
   - **Mitigation:** Automatic scrubbing for core regulatory components (Aadhaar, PAN, UPI, credit cards, standard API keys).

4. **Resource Exhaustion (Application DoS)**
   - **Vector:** Pumping thousands of NFD-unicode expansion characters, newline token injections, or recursive single-word payload bombs into the prompt to crash standard parsing frameworks.
   - **Mitigation:** First-line `DoSGuard` intercepts malformed boundaries instantly before hitting heavy ML models.

---

## 🚫 Threats NOT Handled (Out-of-Scope)

To build architectural trust, it is critical to explicitly note what this system *does not* do.

1. **LLM Hallucinations / Factual Inaccuracy**
   - Kavach analyzes structural intent and tool execution risk. If a model generates a false, yet benign statement regarding an internal policy or calculation, Kavach will not intercept it. Factual grounding is a domain belonging to the Retrieval Engine (RAG), not the security gateway.

2. **Authorized Insider Misuse (Policy Misconfiguration)**
   - If an Administrator grants the `delete_production_db` tool to the `guest_user` role in the `policy.yaml`, Kavach will reliably execute the delete command. We enforce policy; we do not govern the logic of the human who writes the policy.

3. **Adversarial Token Tuning (Mathematical Optimization)**
   - Soft-prompt token optimization, where an adversary calculates precise, seemingly random strings of letters (e.g., "Zjg %^1...") that exploit particular gradient pathways in an open-weights model, currently lie below the line of deterministic semantic recognition. The ML embeddings mitigate this partially, but complete coverage requires parameter-level model alignment modifications, separate from the gateway.

---

## ⚠️ Failure Modes & Graceful Degradation

Since Kavach acts as a primary application conduit, it is designed around the principle of **Graceful Degradation** rather than hard systemic crashing.

*   **Redis Offline:** If the shared state behavioral backend drops, Kavach falls back dynamically to a stateless request tracker. It will not remember cross-pod behavior for the duration of the outage, but it will seamlessly keep processing risk assessments.
*   **External KMS Offline:** If the Key Management Server delivering the HMAC-SHA256 rotating secret becomes unreadable, Kavach falls back to standard memory ENV secrets and tags logs with a `KMS_FAIL` warning. Evaluation availability is prioritized over ideal cryptographic rotating states.
*   **Machine Learning Engine Timeout / OOM:** If the ONNX scoring layer errors out due to unexpected resource utilization or extreme payload conditions, Kavach suppresses the ML signals and falls back to a deterministic rule-based evaluation (100% safe path computation). It logs an ML execution error but allows valid policy requests to proceed unimpeded.
