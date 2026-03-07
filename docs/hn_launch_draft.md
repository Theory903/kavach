**Title:** Show HN: Kavach — Firewall for LLM Apps (Prompt Injection Protection)

**Content:**
Hey HN,

I'm building Kavach (https://github.com/theory903/kavach), an open-source security gateway for LLM applications that protects against prompt injection, jailbreaks, and unauthorized agent actions.

As we started building Agentic AI systems with tool access (reading databases, hitting internal APIs), we realized that "system prompt instructions" weren't enough to stop attacks. We tried the "LLM-as-a-judge" approach to evaluate prompts before processing them, but adding 2 seconds of latency to every interaction completely ruined the user experience and doubled our API costs.

Kavach solves this at the infrastructure level. It sits as a proxy between your app and the LLM (OpenAI, Anthropic, local vLLM). 

Here is how it works under the hood:
1. **Sub-15ms ML Ensemble:** It evaluates incoming prompts using an offline ensemble (GBM, Logistic Regression, etc.) for structural attack signatures. No LLM bottleneck—the p95 overhead is under 15ms.
2. **Tool / RBAC Firewall:** It strictly enforces Role-Based Access Control via a `policy.yaml`. If an attacker tricks a "read-only analyst" agent into executing a destructive tool, the gateway blocks it before the model processes it.
3. **Data Redaction:** It scrubs PII and AWS secrets before the prompt reaches external cloud providers.
4. **Audit Chain:** Cryptographic non-repudiation using HMAC-SHA256 for all decisions.

It's designed as a native drop-in wrapper for existing OpenAI/Anthropic SDKs or LangChain, or it can be run as a standalone container proxy.

The code is Apache 2.0 and we're looking for feedback on the architecture, the offline models, and the roadmap. Would love to hear your thoughts or see if this solves problems you're facing with GenAI deployments in prod.

GitHub: https://github.com/theory903/kavach
