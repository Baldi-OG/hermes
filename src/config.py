# added to git since no secrets
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(override=True)


class LLMBackend:
    # Webis hosted LLM services (requires VPN)
    # https://kb.webis.de/services/llm-staging/index.html#openai-api-vpn-only-unauthenticated
    WEBIS_URL = "https://llm.srv.webis.de/openai/v1/"
    WEBIS_KEY = os.getenv("OLLAMA_KEY")
    # https://llm.srv.webis.de/openai/v1/models
    WEBIS_MODEL = "qwen3-30b-a3b"   # Number of Parameters: 30.5B in total and 3.3B activated (https://huggingface.co/Qwen/Qwen3-30B-A3B)
    WEBIS = "webis"

    # TODO: @students: use the one below
    # Webis hosted LLM services (requires OpenWebUI API key)
    # https://kb.webis.de/services/llm-staging/index.html#openai-api-vpn-only-unauthenticated
    WEBIS_URL_WEBUI = "https://chat.web.webis.de/openai/"
    WEBIS_KEY_WEBUI = os.getenv("OPENWEBUI_WEBIS_KEY")
    # https://llm.srv.webis.de/openai/v1/models
    WEBIS_MODEL_WEBUI = "qwen3-30b-a3b"   # Number of Parameters: 30.5B in total and 3.3B activated (https://huggingface.co/Qwen/Qwen3-30B-A3B)
    WEBIS_WEBUI = "webis"

    # Blablador 
    BLABLADOR_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/ "
    BLABLADOR_KEY = os.getenv("BLABLADOR_KEY")
    BLABLADOR_MODEL = "alias-fast" # "02 - Qwen3.5-122B-A10B-FP8"
    BLABLADOR = "blablador"

    # OpenAI
    OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
    OPENAI_URL = "https://api.openai.com/v1"
    OPENAI_MODEL = "openai/gpt-5-nano-2025-08-07"  # specify snapshot for consistency: https://platform.openai.com/docs/models/gpt-5-nano (30.10.2025)
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    OPENAI = "openai"

CONFIG = LLMBackend()
