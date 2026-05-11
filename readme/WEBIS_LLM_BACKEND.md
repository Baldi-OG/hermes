# Using Custom LLM Backend in VS Code Copilot Chat hosted by Webis

This document explains how to use Webis-hosted LLMs in your workflow.

Using Webis LLMs directly in VS Code Copilot Chat currently does not work reliably.
Instead, you should use one of the following approaches:

- VS Code plugin Continue (recommended for editor integration)
- A Python library (e.g., smolagents, etc.)
---

## Access

The Webis LLM API is accessible via [VPN](https://kb.webis.de/services/llm-staging/index.html#openai-api-vpn-only-unauthenticated) or OpenWebUI API key.

---

## API Endpoint

[Available models](https://chat.web.webis.de/openai/models) are adressable via the OpenAI API.

### For VPN:
Webis provides an OpenAI-compatible API:
````https://llm.srv.webis.de/openai/v1/````
- Webis' [knowledge base](https://kb.webis.de/services/llm-staging/index.html#webis-llms) for instructions on how to connect to Webis hosted LLMs.

### With OpenWebUI API Key
Webis provides an OpenAI-compatible API:
````https://chat.web.webis.de/openai/````
- Webis' [knowledge base](https://kb.webis.de/services/llm-staging/index.html#webis-llms) for instructions on how to connect to Webis hosted LLMs.


---

### Option A: `n8n`/ `Langchain`

A flexible and scalable approach is to use orchestration frameworks like **n8n** or **LangChain**.

These tools let you move beyond simple prompt-response setups and build structured, multi-step AI workflows. You can connect LLMs to APIs, databases, and custom logic—making them ideal for real applications like automation, data pipelines, or agent-based systems.

* **n8n** is great if you prefer a visual, low-code environment. It allows you to quickly wire together triggers, actions, and LLM calls without writing much code.
* **LangChain** is better suited for programmatic control, offering fine-grained abstractions for chains, tools, memory, and agents.

Both integrate cleanly with Webis-hosted models and give you:

* More control over prompt flows and tool usage
* Easier scaling and reuse of components
* A clear path from prototype to production

Compared to simple plugins or local setups, this option is more extensible, maintainable, and powerful—especially if you're building anything beyond basic chat interactions.


### Option B: `Continue` Plugin (Do not use)
Since Copilot Chat does not work with this backend, use `Continue` instead.
Edit `/Users/<name>/.continue/config.yaml`:
````
name: Webis Local
version: 0.0.1
schema: v1

models:
  - name: Webis Qwen Chat
    provider: openai
    model: qwen3-30b-a3b
    apiBase: https://llm.srv.webis.de/openai/v1
    apiKey: ${env:WEBIS_KEY}
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use
````

Create a `.env` file.
Change the apiBase if you use an API key.
It should store your Webis API key (*if not using VPN*) `WEBIS_KEY=<key>`. If you do not have a key, use a placeholder dummy and only access via VPN.


### Option C: Python Library `smolagents` (Do not use)
If you are working in Python, you can directly use Webis-hosted LLMs via libraries such as [`smolagents`](https://github.com/huggingface/smolagents). In this case, using the VS Code Continue plugin is not necessary.

Example configuration:
````
from config import CONFIG
from smolagents import OpenAIServerModel
model = OpenAIServerModel(
    model_id="qwen3-30b-a3b",
    api_base=CONFIG.WEBIS_URL,
    api_key=CONFIG.WEBIS_KEY,
)
````
Using Webis-hosted models enables access to significantly larger models compared to local solutions like Ollama, which are constrained by your machine's available RAM/VRAM.