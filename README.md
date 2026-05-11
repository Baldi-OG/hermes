# Forensic Linguist Agent

## Overview

This project implements a **forensic linguistics agent** designed to analyze **idiolectal writing style**—that is, the unique linguistic patterns of an individual—**independent of text content or topic**.

The system is built as a (multi-)agent architecture inside VS Code, combining:

* A planning and coordination agent
* Multiple tools
* A private LLM backend

The goal is to support structured, explainable forensic text analysis workflows that can be used for tasks such as authorship analysis, stylistic comparison, and evidence assessment.

---

## Agentic Behaviour

This project demonstrates agentic behaviour: instead of executing fixed code paths, the system can dynamically decide what to do next based on a user request.

Differences (WIP/TODO):
- An **agent** runs a tool in a loop to achieve a goal. It calls a LLM, passes it a set of tool defintions, calls tools the LLM requests and feeds back the results [[reference]](https://simonwillison.net/guides/agentic-engineering-patterns/what-is-agentic-engineering/).
- A **tool** is function made available to the LLM [[reference]](https://simonwillison.net/guides/agentic-engineering-patterns/what-is-agentic-engineering/).
- A **skill** is a collection of scripts, instructions and resources [[reference]](https://medium.com/@tahirbalarabe2/what-are-agent-skills-c7793b206daf); is a tool.

---

## Architecture

### Main Agent 

The central agent is responsible for:

* TODO


### Tools

Each tool encapsulates a (specific linguistic analysis) capability, for example:

* N-gram / lexical pattern analysis
* Stylistic feature extraction
* Structural and syntactic analysis
* TODO


### Analysis Focus: Idiolect

The system explicitly focuses on:

> **How something is written, not what is written**

This includes:

* lexical preferences
* grammatical constructions
* punctuation habits
* stylistic consistency

Content (topic, meaning) is treated as **irrelevant noise** unless required for interpretation.

---

## Get Started

This project uses a **Dev Containers**.

👉 See details here:
**[DEVCONTAINER.md](./readme/DEVCONTAINER.md)**

You need to create a `.env` file to store your API keys (the variable names are given in `config.py`).
You will receive a OpenWebUI API key from your superviser.
Do **NOT** push your `.env` file.


---

## LLM Backend

This project uses a **privately hosted LLM backend** instead of public APIs.
You set up both connection to Webis infrastructure or your own Ollama model instances.

👉 See details here:
- **[Local Ollama Backend](./readme/OLLAMA_BACKEND.md)**
- **[Webis-hosted LLM Backend](./readme/WEBIS_LLM_BACKEND.md)**

Store your API keys in an `.env` file (e.g., `BLABLADOR_KEY="<API-KEY>"`).

That document explains:

* integration with VS Code Copilot
* final working setup 
  * using Continue (Webis-hosted LLM Backend)
  * using Python library (Webis-hosted LLM Backend)
  * VSCode AI chat (Local Ollama Backend)
* environment and configuration details


---

## Tech Stack

* VS Code (Dev Container)
* Continue extension (LLM interface)
* OpenAI-compatible API (Webis backend)
* YAML-based model configuration
* Python (analysis tooling)


---

## Design Principles

* **Separation of concerns**
  Planning, execution, and analysis are split across agents

* **Reproducibility**
  Analyses follow structured, repeatable steps

* **Model independence**
  Backend is abstracted via OpenAI-compatible interface

* **Forensic rigor**
  Emphasis on:

  * evidence vs interpretation
  * uncertainty handling
  * transparency of reasoning

---

## Next Steps

* Come up with model of workflow
    * Which components are required?
* Use `n8n`/ `LangChain` as orchestration framework
* Expand linguistic feature set
  * Define skills/ tools
* Implement tools
* Add uncertainty quantification
* Formalize reporting for forensic use cases
* Compare Webis and [Blablador](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/) hosted LLMs as backend
  * You can check whether your API key `<API-KEY>` works by running the cli command `curl --header "Authorization: Bearer <API-KEY>" https://api.helmholtz-blablador.fz-juelich.de/v1/models`

---

## Summary

This project provides a foundation for building structured, agent based forensic linguistics systems powered by a private LLM backend, with a strong focus on idiolect analysis and methodological transparency.

