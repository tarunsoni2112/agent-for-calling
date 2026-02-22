from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = ROOT / "negotiation_system_prompt.txt"
WRAPPER_TEMPLATE_PATH = ROOT / "request_wrapper_template.txt"
MODEL_SETTINGS_PATH = ROOT / "model_settings.json"


@dataclass(frozen=True)
class NegotiationContext:
    name: str
    requirement: str
    language_mode: str
    min_price: float
    ideal_price: float
    walkaway_price: float
    summary: str
    latest_user_input: str


class ContextValidationError(ValueError):
    pass


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_system_prompt() -> str:
    return _read_text(SYSTEM_PROMPT_PATH)


def load_wrapper_template() -> str:
    return _read_text(WRAPPER_TEMPLATE_PATH)


def load_model_settings() -> dict[str, Any]:
    with MODEL_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "temperature" not in data:
        raise ContextValidationError("model_settings.json missing 'temperature'")
    return data


def validate_context(context: NegotiationContext) -> None:
    allowed_languages = {"Hindi", "Hinglish", "Marathi"}
    if context.language_mode not in allowed_languages:
        raise ContextValidationError(
            f"Unsupported language_mode: {context.language_mode}. "
            f"Allowed: {', '.join(sorted(allowed_languages))}"
        )

    floor_price = max(context.min_price, context.walkaway_price)
    if context.ideal_price < floor_price:
        raise ContextValidationError(
            "ideal_price cannot be below max(min_price, walkaway_price)"
        )


def build_user_message(context: NegotiationContext) -> str:
    validate_context(context)
    template = load_wrapper_template()

    replacements = {
        "{{name}}": context.name,
        "{{requirement}}": context.requirement,
        "{{language_mode}}": context.language_mode,
        "{{min}}": _format_price(context.min_price),
        "{{ideal}}": _format_price(context.ideal_price),
        "{{walkaway}}": _format_price(context.walkaway_price),
        "{{summary}}": context.summary,
        "{{latest_user_input}}": context.latest_user_input,
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_payload(context: NegotiationContext) -> dict[str, Any]:
    settings = load_model_settings()
    return {
        "system_prompt": load_system_prompt(),
        "user_message": build_user_message(context),
        "model_settings": settings,
    }


def _format_price(price: float) -> str:
    if float(price).is_integer():
        return str(int(price))
    return f"{price:.2f}"
