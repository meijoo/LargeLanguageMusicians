"""Environment configuration and an Azure v1 Responses adapter for AutoGen."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from openai import OpenAI

if __package__:
    from .composer_protocol import responses_text_format
else:
    from composer_protocol import responses_text_format


DEFAULT_MODEL = "gpt-6.1-luna"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


def _azure_base_url(endpoint: str) -> str:
    # Accept an origin or an already-normalized v1 URL, without silently
    # appending the v1 path twice or retaining a legacy deployment route.
    parts = urlsplit(endpoint)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment
            or parts.path.rstrip("/") not in ("", "/openai/v1")):
        raise ValueError("OPENAI_ENDPOINT/AZURE_OPENAI_ENDPOINT must be an HTTP(S) origin "
                         "or an /openai/v1/ URL, without credentials, query, or fragment.")
    return urlunsplit((parts.scheme, parts.netloc, "/openai/v1/", "", ""))


def build_llm_config(
    model: Optional[str] = None, *, env_file: Optional[Path] = None,
) -> tuple[dict, str]:
    """Resolve credentials without exporting or logging them.

    Shell variables override .env; Azure aliases override generic aliases;
    an explicit model argument overrides both deployment/model environment names.
    """
    load_dotenv(env_file if env_file is not None else PROJECT_ROOT / ".env", override=False)
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a nonempty string.")
    api_key = _environment("AZURE_OPENAI_API_KEY") or _environment("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env or the shell.")
    endpoint = _environment("AZURE_OPENAI_ENDPOINT") or _environment("OPENAI_ENDPOINT")
    base_url = _environment("OPENAI_BASE_URL")
    resolved_model = ((model.strip() if model is not None else None)
                      or _environment("AZURE_OPENAI_DEPLOYMENT") or _environment("OPENAI_MODEL"))
    config = {"api_key": api_key}
    if endpoint:
        if not resolved_model:
            raise ValueError("Set OPENAI_MODEL or AZURE_OPENAI_DEPLOYMENT to your enabled "
                             "deployment alias, or supply --model.")
        config.update(base_url=_azure_base_url(endpoint), model_client_cls="ResponsesModelClient")
    else:
        azure_settings = _environment("AZURE_OPENAI_API_KEY") or _environment("AZURE_OPENAI_DEPLOYMENT")
        if azure_settings:
            if not base_url:
                raise ValueError("Azure settings require AZURE_OPENAI_ENDPOINT or OPENAI_ENDPOINT.")
            if not resolved_model:
                raise ValueError("Set OPENAI_MODEL or AZURE_OPENAI_DEPLOYMENT to your enabled "
                                 "deployment alias, or supply --model.")
        if base_url:
            config["base_url"] = base_url
        resolved_model = resolved_model or DEFAULT_MODEL
    config["model"] = resolved_model
    return {
        "config_list": [config], "cache_seed": None, "timeout": 120, "max_retries": 2,
    }, resolved_model


class ResponsesModelClient:
    """Adapt text Responses API calls to AutoGen 0.2's model-client protocol.

    Azure v1 uses the standard OpenAI client. The explicit api-key header also
    supports gateways that expect Azure key authentication. Cost is unestimated;
    token usage is retained, while billing remains with the provider.
    """

    def __init__(self, config: dict):
        self.model = config["model"]
        self.client = OpenAI(
            api_key=config["api_key"], base_url=config["base_url"],
            timeout=config.get("timeout", 120), max_retries=config.get("max_retries", 2),
            http_client=config.get("http_client"),
            default_headers={"api-key": config["api_key"]},
        )

    def create(self, params: dict) -> SimpleNamespace:
        messages = []
        for message in params["messages"]:
            if (message.get("role") not in ("system", "developer", "user", "assistant")
                    or not isinstance(message.get("content"), str)):
                raise ValueError("The composer Responses client accepts text messages only.")
            messages.append({"role": message["role"], "content": message["content"]})
        response = self.client.responses.create(
            model=params.get("model", self.model), input=messages, store=False,
            text={"format": responses_text_format()},
        )
        if response.status != "completed":
            raise ValueError("The Responses API did not complete the composition.")
        text = response.output_text
        if not text or not text.strip():
            raise ValueError("The Responses API returned no composition text.")
        usage = response.usage
        return SimpleNamespace(
            model=response.model,
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(
                prompt_tokens=usage.input_tokens if usage else 0,
                completion_tokens=usage.output_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    def message_retrieval(self, response: SimpleNamespace) -> list[str]:
        return [choice.message.content for choice in response.choices]

    def cost(self, response: SimpleNamespace) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: SimpleNamespace) -> dict:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": 0.0,
            "model": response.model,
        }
