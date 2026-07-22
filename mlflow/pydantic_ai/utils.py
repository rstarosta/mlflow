"""Helpers shared by the Pydantic AI 1.x and 2.x autologging implementations."""

import inspect
import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

_logger = logging.getLogger(__name__)
_SAFE_PRIMITIVE_TYPES = (str, int, float, bool)


def _is_safe_for_serialization(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, _SAFE_PRIMITIVE_TYPES):
        return True
    if isinstance(value, dict):
        return all(_is_safe_for_serialization(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_safe_for_serialization(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return True
    if isinstance(value, type):
        return True
    return False


def _safe_get_attribute(instance: Any, key: str) -> Any:
    try:
        value = getattr(instance, key, None)
        if value is None:
            return None
        if isinstance(value, type):
            return value.__name__
        if _is_safe_for_serialization(value):
            return value
        return None
    except Exception:
        return None


def _extract_safe_attributes(instance: Any) -> dict[str, Any]:
    """Extract public attributes that are safe for serialization."""
    attrs = {}
    for key in dir(instance):
        if key.startswith("_"):
            continue
        value = getattr(instance, key, None)
        if callable(value) and not isinstance(value, type):
            continue
        safe_value = _safe_get_attribute(instance, key)
        if safe_value is not None:
            attrs[key] = safe_value
    return attrs


def _construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs).arguments
        bound.pop("self", None)
        bound.pop("deps", None)

        return {
            k: (v.__dict__ if hasattr(v, "__dict__") else v)
            for k, v in bound.items()
            if v is not None
        }
    except (ValueError, TypeError):
        return kwargs


def _serialize_output(result: Any) -> Any:
    if result is None:
        return None

    if hasattr(result, "new_messages") and callable(result.new_messages):
        try:
            new_messages = result.new_messages()
            serialized_messages = [asdict(msg) for msg in new_messages]

            try:
                serialized_result = asdict(result)
            except Exception:
                serialized_result = dict(result.__dict__) if hasattr(result, "__dict__") else {}

            serialized_result["_new_messages_serialized"] = serialized_messages
            return serialized_result
        except Exception as e:
            _logger.debug("Failed to serialize new_messages: %s", e)

    return result.__dict__ if hasattr(result, "__dict__") else result


def _get_agent_attributes(instance):
    attrs = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attrs.update(_extract_safe_attributes(instance))
    if hasattr(instance, "tools"):
        try:
            if tools_value := _parse_tools(instance.tools):
                attrs["tools"] = tools_value
        except Exception:
            pass
    return attrs


def _get_model_attributes(instance):
    attrs = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attrs.update(_extract_safe_attributes(instance))
    return attrs


def _get_tool_attributes(instance):
    return _extract_safe_attributes(instance)


def _get_toolset_attributes(instance):
    attrs = _extract_safe_attributes(instance)
    if hasattr(instance, "tools"):
        try:
            if tools_value := _parse_tools(instance.tools):
                attrs["tools"] = tools_value
        except Exception:
            pass
    return attrs


def _parse_tools(tools):
    return [
        {"type": "function", "function": data}
        for tool in tools
        if (data := tool.model_dumps(exclude_none=True))
    ]


def _parse_usage(result: Any) -> dict[str, int] | None:
    try:
        if isinstance(result, tuple) and len(result) == 2:
            usage = result[1]
        else:
            usage_attr = getattr(result, "usage", None)
            if usage_attr is None:
                return None
            usage = usage_attr() if callable(usage_attr) else usage_attr

        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "request_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "response_tokens", 0)
        total_tokens = getattr(usage, "total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return {
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: total_tokens,
        }
    except Exception as e:
        _logger.debug("Failed to parse token usage from output: %s", e)
    return None


def _model_request_inputs(request_context) -> dict[str, Any]:
    if request_context is None:
        return {}
    inputs = {
        "messages": getattr(request_context, "messages", None),
        "model_settings": getattr(request_context, "model_settings", None),
        "model_request_parameters": getattr(request_context, "model_request_parameters", None),
    }
    return {k: v for k, v in inputs.items() if v is not None}
