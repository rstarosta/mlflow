"""Autologging implementation for Pydantic AI 2.x."""

import contextvars
import functools
import inspect
import logging
import typing
from contextlib import asynccontextmanager

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.pydantic_ai.utils import (
    _construct_full_inputs,
    _get_agent_attributes,
    _get_model_attributes,
    _get_tool_attributes,
    _get_toolset_attributes,
    _model_request_inputs,
    _parse_usage,
    _serialize_output,
)
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import with_active_span
from mlflow.utils.autologging_utils import safe_patch
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _store_patch, _wrap_patch

_logger = logging.getLogger(__name__)
_in_sync_stream_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_sync_stream_context", default=False
)


def _set_span_attributes(span: LiveSpan, instance):
    try:
        from pydantic_ai.mcp import MCPToolset

        if isinstance(instance, MCPToolset):
            attrs = _get_toolset_attributes(instance)
            span.set_attributes({k: v for k, v in attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving MCPToolset attributes: %s", e)

    try:
        from pydantic_ai import Agent

        if isinstance(instance, Agent):
            attrs = _get_agent_attributes(instance)
            span.set_attributes({k: v for k, v in attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Agent attributes: %s", e)

    try:
        from pydantic_ai.models import Model

        if isinstance(instance, Model):
            attrs = _get_model_attributes(instance)
            span.set_attributes({k: v for k, v in attrs.items() if v is not None})
            if model_name := getattr(instance, "model_name", None):
                span.set_attribute(SpanAttributeKey.MODEL, model_name)
                provider = getattr(instance, "system", None)
                if provider is None:
                    match model_name.split(":", 1):
                        case [prefix, _]:
                            provider = prefix
                if provider:
                    span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, provider)
    except Exception as e:
        _logger.warning("Failed saving model attributes: %s", e)

    try:
        from pydantic_ai import Tool

        if isinstance(instance, Tool):
            attrs = _get_tool_attributes(instance)
            span.set_attributes({k: v for k, v in attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Tool attributes: %s", e)


def patched_agent_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if cfg.log_traces and self.instrument is None:
        self.instrument = True
    return result


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    with mlflow.start_span(name=fullname, span_type=_get_span_type(self)) as span:
        span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
        _set_span_attributes(span, self)

        result = await original(self, *args, **kwargs)
        span.set_outputs(_serialize_output(result))
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


async def patched_capability_model_request(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    request_context = kwargs.get("request_context")
    if request_context is None and args:
        try:
            from pydantic_ai.models import ModelRequestContext

            request_context = next((a for a in args if isinstance(a, ModelRequestContext)), None)
        except ImportError:
            request_context = None
    model = getattr(request_context, "model", None)

    span_name = f"{type(model).__name__}.request" if model is not None else "Model.request"
    with mlflow.start_span(name=span_name, span_type=SpanType.LLM) as span:
        span.set_inputs(_model_request_inputs(request_context))
        if model is not None:
            _set_span_attributes(span, model)

        result = await original(self, *args, **kwargs)
        span.set_outputs(_serialize_output(result))
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


def patched_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    with mlflow.start_span(name=fullname, span_type=_get_span_type(self)) as span:
        span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
        _set_span_attributes(span, self)

        result = original(self, *args, **kwargs)
        span.set_outputs(_serialize_output(result))
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


def patched_async_stream_call(original, self, *args, **kwargs):
    @asynccontextmanager
    async def _wrapper():
        cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
        if not cfg.log_traces:
            async with original(self, *args, **kwargs) as result:
                yield result
            return

        from pydantic_ai import Agent

        if _in_sync_stream_context.get() and isinstance(self, Agent):
            async with original(self, *args, **kwargs) as result:
                yield result
            return

        fullname = f"{self.__class__.__name__}.{original.__name__}"
        with mlflow.start_span(name=fullname, span_type=_get_span_type(self)) as span:
            span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
            _set_span_attributes(span, self)

            async with original(self, *args, **kwargs) as stream_result:
                try:
                    yield stream_result
                finally:
                    try:
                        span.set_outputs(_serialize_output(stream_result))
                        if usage_dict := _parse_usage(stream_result):
                            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
                    except Exception as e:
                        _logger.debug("Failed to set streaming outputs: %s", e)

    return _wrapper()


class _StreamedRunResultSyncWrapper:
    def __init__(self, result, span):
        self._result = result
        self._span = span
        self._finalized = False
        self._closed = False

    def _use_span_context(self):
        return with_active_span(self._span)

    def _close_result(self, exc_type=None, exc_val=None, exc_tb=None):
        if self._closed:
            return None
        self._closed = True
        with self._use_span_context():
            return self._result.__exit__(exc_type, exc_val, exc_tb)

    def _finalize(self, exc_type=None, exc_val=None, exc_tb=None):
        if self._finalized:
            return
        self._finalized = True

        try:
            self._close_result(exc_type, exc_val, exc_tb)
            self._span.set_outputs(_serialize_output(self._result))
            if usage_dict := _parse_usage(self._result):
                self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        except Exception as e:
            _logger.debug("Failed to set streaming outputs: %s", e)
        finally:
            if exc_type is not None:
                self._span.end(status="ERROR")
            else:
                self._span.end()

    def _wrap_iterator(self, iterator_func, **kwargs):
        with self._use_span_context():
            try:
                yield from iterator_func(**kwargs)
            finally:
                self._finalize()

    def stream_text(self, **kwargs):
        return self._wrap_iterator(self._result.stream_text, **kwargs)

    def stream_output(self, **kwargs):
        return self._wrap_iterator(self._result.stream_output, **kwargs)

    def stream_responses(self, **kwargs):
        return self._wrap_iterator(self._result.stream_responses, **kwargs)

    def get_output(self, **kwargs):
        with self._use_span_context():
            try:
                return self._result.get_output(**kwargs)
            finally:
                self._finalize()

    def __enter__(self):
        self._result.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        return getattr(self._result, name)


def patched_sync_stream_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span = mlflow.start_span_no_context(name=fullname, span_type=_get_span_type(self))
    span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
    _set_span_attributes(span, self)

    try:
        token = _in_sync_stream_context.set(True)
        try:
            with with_active_span(span):
                result = original(self, *args, **kwargs)
        finally:
            _in_sync_stream_context.reset(token)

        return _StreamedRunResultSyncWrapper(result, span)
    except Exception:
        span.end(status="ERROR")
        raise


def _get_span_type(instance) -> str:
    try:
        from pydantic_ai.models import Model

        if isinstance(instance, Model):
            return SpanType.LLM
    except ImportError:
        pass

    try:
        from pydantic_ai import Agent

        if isinstance(instance, Agent):
            return SpanType.AGENT
    except ImportError:
        pass

    try:
        from pydantic_ai import Tool

        if isinstance(instance, Tool):
            return SpanType.TOOL
    except ImportError:
        pass

    try:
        from pydantic_ai.mcp import MCPToolset

        if isinstance(instance, MCPToolset):
            return SpanType.TOOL
    except ImportError:
        pass

    try:
        from pydantic_ai.tool_manager import ToolManager

        if isinstance(instance, ToolManager):
            return SpanType.TOOL
    except ImportError:
        pass

    return SpanType.UNKNOWN


def _is_async_context_manager_factory(func) -> bool:
    wrapped = getattr(func, "__wrapped__", None)
    return wrapped is not None and inspect.isasyncgenfunction(wrapped)


def _returns_sync_streamed_result(func) -> bool:
    if inspect.iscoroutinefunction(func):
        return False

    try:
        return_annotation = inspect.signature(func).return_annotation
    except (ValueError, TypeError):
        return False

    if return_annotation is inspect.Signature.empty:
        return False
    if isinstance(return_annotation, str):
        return "StreamedRunResultSync" in return_annotation

    origin = typing.get_origin(return_annotation) or return_annotation
    return hasattr(origin, "stream_text") and hasattr(origin, "stream_output")


def _patch_streaming_method(cls, method_name, wrapper_func):
    original = getattr(cls, method_name)

    @functools.wraps(original)
    def patched_method(self, *args, **kwargs):
        return wrapper_func(original, self, *args, **kwargs)

    patch = _wrap_patch(cls, method_name, patched_method)
    _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)


def _patch_method(cls, method_name):
    method = getattr(cls, method_name)
    if _is_async_context_manager_factory(method):
        _patch_streaming_method(cls, method_name, patched_async_stream_call)
    elif _returns_sync_streamed_result(method):
        _patch_streaming_method(cls, method_name, patched_sync_stream_call)
    elif inspect.iscoroutinefunction(method):
        safe_patch(mlflow.pydantic_ai.FLAVOR_NAME, cls, method_name, patched_async_class_call)
    else:
        safe_patch(mlflow.pydantic_ai.FLAVOR_NAME, cls, method_name, patched_class_call)


def setup_autologging() -> None:
    """Install the Pydantic AI 2.x autologging patches."""
    from pydantic_ai import Agent

    agent_methods = ["run", "run_sync", "run_stream"]
    if hasattr(Agent, "run_stream_sync"):
        agent_methods.append("run_stream_sync")

    class_map = {
        "pydantic_ai.Agent": agent_methods,
        "pydantic_ai.tool_manager.ToolManager": ["execute_tool_call"],
        "pydantic_ai.mcp.MCPToolset": ["call_tool", "get_tools"],
    }

    original_init = Agent.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        return patched_agent_init(original_init, self, *args, **kwargs)

    patch = _wrap_patch(Agent, "__init__", patched_init)
    _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)

    for cls_path, methods in class_map.items():
        module_name, class_name = cls_path.rsplit(".", 1)
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            _logger.error("Error importing %s: %s", cls_path, e)
            continue

        for method in methods:
            try:
                _patch_method(cls, method)
            except AttributeError as e:
                _logger.error("Error patching %s.%s: %s", cls_path, method, e)

    try:
        from pydantic_ai.capabilities.instrumentation import Instrumentation

        safe_patch(
            mlflow.pydantic_ai.FLAVOR_NAME,
            Instrumentation,
            "wrap_model_request",
            patched_capability_model_request,
        )
    except (ImportError, AttributeError) as e:
        _logger.error("Error patching Instrumentation.wrap_model_request: %s", e)
