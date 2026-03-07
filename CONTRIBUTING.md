# Contributing to Kavach 🛡️

First off, thank you for considering contributing to Kavach! It's people like you that make Kavach an enterprise-grade, zero-trust security gateway. 

We are an open-source security project, which means community trust, code quality, and performance (sub-15ms latency!) are our top priorities. 

This document provides a clear handbook on how to contribute successfully.

---

## 🗺️ What should I work on?

If you're looking for a place to start, check out the [Issues tab](https://github.com/theory903/kavach/issues) and look for the following labels:
- `good first issue`: Ideal for getting familiar with the codebase.
- `help wanted`: Features or bugs we specifically need community help with.
- `ml-engine`: Tasks related to our offline ML ensemble.
- `integrations`: Adding support for more LLM frameworks (like Haystack, AutoGen).

If you want to propose a **massive architectural change**, please open a GitHub Discussion or an Issue *first* so we can align on the design before you spend hours coding!

---

## 🛠️ Local Development Setup

Kavach requires **Python 3.10+**. We use `poetry` (or standard `pip` with virtual environments) for dependency management.

### 1. Fork and Clone
1. Fork this repository to your own GitHub account.
2. Clone it locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kavach.git
   cd kavach
   ```

### 2. Install Dependencies
Set up your virtual environment and install both the core requirements and the test requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,ml]"
```

### 3. Run the Tests
We maintain high test coverage. Before writing new code, ensure the current test suite passes:
```bash
pytest tests/
```

---

## ✍️ Development Guidelines

To keep the codebase maintainable and lightning-fast:

### Code Style (PEP 8)
- We use **Black** for formatting.
- We use **Ruff** (or Flake8) for linting.
- We enforce strong **Type Hinting** for all function signatures.

Run the formatter before committing:
```bash
black kavach/ tests/
```

### Latency Budget (CRITICAL)
Kavach is designed to have a p95 overhead of `<15ms`. If you are adding a feature to the `gateway.py` or the request interception path:
- **Do not introduce blocking network calls** (e.g., synchronous HTTP requests).
- **Optimize ML feature extraction**.
- Use asynchronous operations (`asyncio`) where necessary.

---

## 🚢 Submitting a Pull Request (PR)

1. **Branch Naming:** Create a new branch from `main`. Use a descriptive name like `feat/redis-caching` or `fix/prompt-injection-regex`.
2. **Commit Messages:** We follow semantic commit messages (e.g., `feat: add Anthropic integration` or `fix: resolve KMS latency bug`).
3. **Write Tests:** If you add a new integration or rule, add a corresponding test in the `tests/` directory.
4. **Open the PR:** Fill out the Pull Request template provided by GitHub. Please include screenshots or benchmark logs if applicable.

### The Review Process
One of our core maintainers will review your PR within 48 hours. We might ask for some changes regarding edge cases, security implications, or latency constraints. Don't be discouraged! This is normal for security infrastructure.

---

## 🔒 Security Vulnerabilities

Since this is a security project, **DO NOT FILE A PUBLIC ISSUE if you discover a critical vulnerability or bypass in Kavach**.
Instead, please immediately refer to our [SECURITY.md](SECURITY.md) guidelines for responsible disclosure.

---

Welcome to the team! Let's make Agentic AI secure together. 🚀
