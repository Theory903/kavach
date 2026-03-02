# Kavach Integrations

Kavach is designed to be completely framework-agnostic. By wrapping execution paths directly, it provides zero-trust security without forcing you to rearchitect your entire LLM pipeline.

Below are the officially supported integration patterns for the most common AI architectures.

---

## 1. OpenAI (Native)

Drop-in replacement for the official `openai` Python package. This pattern intercepts `chat.completions.create` and evaluates the prompt payloads before submission.

```python
from kavach.integrations.openai import SecureOpenAI

# 100% compatible with existing OpenAI client syntax
client = SecureOpenAI(
    api_key="sk-proj-...",
    policy="config/policy.yaml"
)

# Throws KavachBlockedException if prompt is malicious
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Fetch the database records."}],
    user_id="user_123",
    role="analyst" 
)

print(response.choices[0].message.content)
```

---

## 2. Anthropic (Native)

Wraps the official Anthropic client to enforce structural protection and PII redaction on Claude outputs.

```python
from kavach.integrations.anthropic import SecureClaude

client = SecureClaude(policy="config/policy.yaml", user_id="user_123", role="admin")

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summary please."}]
)
```

---

## 3. LangChain

Kavach acts as a deterministic middleware blocking malicious `invoke()` calls within LangChain components.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from kavach.integrations.langchain import SecureChain

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Tell me a story about {topic}")
chain = prompt | llm

# Wrap any chain with Kavach
secure_pipeline = SecureChain(base_chain=chain, policy="config/policy.yaml")

# Run securely
result = secure_pipeline.invoke(
    {"topic": "bears"}, 
    user_id="end_user_88", 
    role="guest"
)
```

---

## 4. LlamaIndex (Upcoming Pattern)

For advanced RAG systems, Kavach can be plugged in as a query transformation or a global callback handler to sanitize contexts before vector search and sanitize generated answers.

*Note: A native LlamaIndex integration is planned for the next minor release. Currently, you can wrap the core LLM via the underlying OpenAI/Anthropic secure clients.*

---

## 5. vLLM / LiteLLM (Proxy Gateway Model)

For organizations running custom weights or router proxies like LiteLLM, Kavach is designed to sit directly in front of the routing layer as a high-speed reverse proxy.

Deploy the FASTAPI server:

```bash
docker run -p 8000:8000 \
  -e KAVACH_POLICY_PATH=/etc/kavach/policy.yaml \
  -e UPSTREAM_LLM_URL=http://internal-vllm:8080/v1/chat/completions \
  ghcr.io/kavach-security/kavach-proxy
```

And point your raw client traffic at the `http://kavach-proxy:8000/v1/secure-run` endpoint. Kavach evaluates risk in < 15ms and streams the completion via the defined Upstream.

---
