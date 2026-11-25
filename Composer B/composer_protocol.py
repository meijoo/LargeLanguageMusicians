"""Shared wire format for full-score composition and peer messages."""

from copy import deepcopy


COMPOSER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "Musical rationale, critique, or concessions."},
        "plan": {
            "type": "array", "items": {"type": "string"},
            "description": "Concrete musical decisions implemented in the complete score.",
        },
        "abc": {"type": "string", "description": "One complete, instrumented multi-voice ABC tune."},
        "accepts_peer_plan": {
            "type": "boolean", "description": "This agent accepts the peer's exact plan; false when composing independently.",
        },
    },
    "required": ["message", "plan", "abc", "accepts_peer_plan"],
    "additionalProperties": False,
}


def responses_text_format() -> dict:
    return {
        "type": "json_schema", "name": "composer_response", "strict": True,
        "schema": deepcopy(COMPOSER_RESPONSE_SCHEMA),
    }


def chat_response_format() -> dict:
    definition = responses_text_format()
    definition.pop("type")
    return {"type": "json_schema", "json_schema": definition}
