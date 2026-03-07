# Security Policy

At Kavach, protecting LLM applications is our entire mission. We take the security of our own framework exceptionally seriously. 

## Supported Versions

We currently provide security updates and patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| v1.0.x  | :white_check_mark: |
| Main branch | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, an orchestration bypass, or an ML model evasion technique within Kavach, **please do not disclose it publicly by opening a GitHub Issue.**

Instead, we ask that you practice responsible disclosure. This gives our maintainers time to patch the vulnerability and protect all users relying on the gateway in production.

**Please report vulnerabilities via email:** [Insert Contact Email / Security Email]

When reporting an issue, please provide:
1. A brief description of the vulnerability.
2. Steps to reproduce the bypass or exploit.
3. If applicable, an example of a malicious prompt that successfully evades the offline ML classifier or RBAC engine. 
4. The version of Kavach you tested against.

### Timeline
- We will acknowledge receipt of your vulnerability report within **48 hours**.
- We will provide an estimated timeline for the patch within **5 business days**.
- Once the patch is merged and released, you may publicly disclose your findings (and you will be properly credited in our release notes for the security discovery!).

## Out of Scope
The following are currently considered accepted risks or out of scope for the Bug Bounty / Security disclosure process:
- Extremely high-token payload Denial of Service (DoS) attacks (these should be handled stringently by the proxy layer API Gateway like Kong/Nginx sitting *in front* of Kavach).
- Vulnerabilities that require the attacker to have direct shell access to the host machine running the Kavach container.
