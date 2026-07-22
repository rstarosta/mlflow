"""Autologging implementation for Pydantic AI 1.x."""

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

# Context variable to track when we're inside run_stream_sync to prevent
# double span creation (run_stream_sync internally calls run_stream)
_in_sync_stream_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_sync_stream_context", default=False
)


def _set_span_attributes(span: LiveSpan, instance):
    # 1) MCPServer attributes
    try:
        from pydantic_ai.mcp import MCPServer

        if isinstance(instance, MCPServer):
            mcp_attrs = _get_toolset_attributes(instance)
            span.set_attributes({k: v for k, v in mcp_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving MCPServer attributes: %s", e)

    # 2) Agent attributes
    try:
        from pydantic_ai import Agent

        if isinstance(instance, Agent):
            agent_attrs = _get_agent_attributes(instance)
            span.set_attributes({k: v for k, v in agent_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Agent attributes: %s", e)

    # 3) Model attributes. `Model` covers both `InstrumentedModel` (pydantic-ai < 1.95)
    #    and the concrete provider models (e.g. `OpenAIChatModel`) that the capabilities-era
    #    request path invokes (pydantic-ai >= 1.95).
    try:
        from pydantic_ai.models import Model

        if isinstance(instance, Model):
            model_attrs = _get_model_attributes(instance)
            span.set_attributes({k: v for k, v in model_attrs.items() if v is not None})
            if model_name := getattr(instance, "model_name", None):
                span.set_attribute(SpanAttributeKey.MODEL, model_name)
                # Prefer the model's own provider identifier (`system`, e.g. "openai").
                # Fall back to the "provider:model" prefix used by some model_name formats
                # (e.g. "anthropic:claude-3-5-haiku"). Only fall back when `system` is
                # genuinely absent (None), not when it's an explicit empty string.
                provider = getattr(instance, "system", None)
                if provider is None:
                    match model_name.split(":", 1):
                        case [prefix, _]:
                            provider = prefix
                if provider:
                    span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, provider)
    except Exception as e:
        _logger.warning("Failed saving model attributes: %s", e)

    # 4) Tool attributes
    try:
        from pydantic_ai import Tool

        if isinstance(instance, Tool):
            tool_attrs = _get_tool_attributes(instance)
            span.set_attributes({k: v for k, v in tool_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Tool attributes: %s", e)


def patched_agent_init(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if cfg.log_traces and kwargs.get("instrument") is None:
        kwargs["instrument"] = True
    return original(self, *args, **kwargs)


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = await original(self, *args, **kwargs)
        outputs = _serialize_output(result)
        span.set_outputs(outputs)
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


async def patched_capability_model_request(original, self, *args, **kwargs):
    """Create an LLM span for the capabilities-era model request path.

    pydantic-ai >= 1.95 no longer routes model calls through
    ``InstrumentedModel.request``/``request_stream``. Instead, every model request
    (streaming and non-streaming) funnels through the ``Instrumentation`` capability's
    ``wrap_model_request``. Unlike the ``InstrumentedModel`` path, the concrete model is
    available on the ``request_context`` argument rather than on ``self`` (which is the
    capability), so we extract it explicitly to type the span and set model attributes.

    A plain span (rather than the ``start_span_no_context`` + finalize-on-completion
    pattern used for ``Agent.run_stream``) is correct for streaming too: pydantic-ai's
    streaming path awaits ``wrap_model_request`` via a cooperative hand-off (the handler
    opens the stream, then blocks until the caller finishes consuming it), so this
    ``await original(...)`` only returns once the stream is fully consumed and yields the
    final ``ModelResponse`` with usage. The span therefore stays open for the whole stream
    and captures outputs + token usage. This is verified by the streaming tests in
    ``tests/pydantic_ai/test_pydanticai_fluent_tracing.py``.
    """
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    request_context = kwargs.get("request_context")
    if request_context is None and args:
        # `request_context` is keyword-only in current pydantic-ai; guard against a
        # positional call on other versions by matching the concrete context type rather
        # than duck-typing (which could pick up an unrelated argument).
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
    span_type = _get_span_type(self)
    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = original(self, *args, **kwargs)
        outputs = _serialize_output(result)
        span.set_outputs(outputs)
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

        # Skip span creation ONLY for Agent.run_stream when inside run_stream_sync.
        # Agent.run_stream_sync already creates a root span, so we don't need another
        # Agent.run_stream span. But we DO still want the nested model-level LLM span,
        # which is created regardless of this skip: from InstrumentedModel.request_stream
        # on pydantic-ai < 1.95, or from the Instrumentation.wrap_model_request capability
        # hook on >= 1.95 (see patched_capability_model_request).
        # The async context manager for Agent.run_stream won't properly exit when
        # called from run_stream_sync (pydantic_ai's implementation uses a generator
        # that pauses), so we skip it to avoid orphaned spans.
        from pydantic_ai import Agent

        if _in_sync_stream_context.get() and isinstance(self, Agent):
            async with original(self, *args, **kwargs) as result:
                yield result
            return

        fullname = f"{self.__class__.__name__}.{original.__name__}"
        span_type = _get_span_type(self)

        with mlflow.start_span(name=fullname, span_type=span_type) as span:
            inputs = _construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            _set_span_attributes(span, self)

            async with original(self, *args, **kwargs) as stream_result:
                try:
                    yield stream_result
                finally:
                    # After the stream is consumed, get the final result
                    try:
                        outputs = _serialize_output(stream_result)
                        span.set_outputs(outputs)
                        if usage_dict := _parse_usage(stream_result):
                            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
                    except Exception as e:
                        _logger.debug(f"Failed to set streaming outputs: {e}")

    return _wrapper()


# Wrapper that captures span outputs after stream is consumed.
# This is necessary because run_stream_sync is NOT a context manager
# (unlike run_stream which is @asynccontextmanager). We must intercept
# iterator completion to know when streaming finishes.
class _StreamedRunResultSyncWrapper:
    def __init__(self, result, span):
        self._result = result
        self._span = span
        self._finalized = False

    def _use_span_context(self):
        return with_active_span(self._span)

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True

        # End child spans that haven't been ended yet.
        # This is necessary because pydantic_ai's run_stream_sync uses an async generator
        # that pauses mid-execution, causing async context managers (and their spans) to
        # never properly exit. We manually end these spans before ending the root span.
        self._end_unfinished_child_spans()

        try:
            self._span.set_outputs(_serialize_output(self._result))
            if usage_dict := _parse_usage(self._result):
                self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        except Exception as e:
            _logger.debug(f"Failed to set streaming outputs: {e}")
        finally:
            self._span.end()

    def _end_unfinished_child_spans(self):
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        manager = InMemoryTraceManager.get_instance()
        if manager is None:
            return

        trace_id = self._span.request_id
        root_span_id = self._span.span_id

        with manager.get_trace(trace_id) as trace:
            if not trace:
                return

            # Find and end all unfinished child spans (direct children of our root span)
            for span_id, span in trace.span_dict.items():
                if span_id == root_span_id:
                    continue
                # Only end spans that are direct children of our root span
                if span.parent_id == root_span_id and span._span.end_time is None:
                    try:
                        span.end()
                    except Exception as e:
                        _logger.debug(f"Failed to end child span {span.name}: {e}")

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

    def __getattr__(self, name):
        return getattr(self._result, name)


def patched_sync_stream_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    # Use start_span_no_context (not `with start_span()`) because the span must remain
    # open after this function returns. The span ends later when the user finishes
    # iterating through the stream (handled by _StreamedRunResultSyncWrapper._finalize).
    span = mlflow.start_span_no_context(name=fullname, span_type=span_type)

    span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
    _set_span_attributes(span, self)

    try:
        # Use use_span to set this span as the active context so child spans
        # (e.g., LLM calls via InstrumentedModel) are properly parented.
        # end_on_exit=False ensures we control when the span ends (in _finalize).
        # Also set _in_sync_stream_context to prevent patched_async_stream_call
        # from creating another Agent.run_stream span (it would never end due to
        # pydantic_ai's async generator implementation).
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
        from pydantic_ai import Agent, Tool
        from pydantic_ai.mcp import MCPServer
        from pydantic_ai.models import Model
    except ImportError:
        return SpanType.UNKNOWN

    # `Model` covers both `InstrumentedModel` (used on pydantic-ai < 1.95) and the
    # concrete provider models (e.g. `OpenAIChatModel`) invoked on the capabilities-era
    # request path (pydantic-ai >= 1.95).
    if isinstance(instance, Model):
        return SpanType.LLM
    if isinstance(instance, Agent):
        return SpanType.AGENT
    if isinstance(instance, Tool):
        return SpanType.TOOL
    if isinstance(instance, MCPServer):
        return SpanType.TOOL

    try:
        from mlflow.pydantic_ai import _get_tool_manager_module_path

        _tm_mod = __import__(_get_tool_manager_module_path(), fromlist=["ToolManager"])
        if isinstance(instance, _tm_mod.ToolManager):
            return SpanType.TOOL
    except ImportError:
        pass
    return SpanType.UNKNOWN


def _get_mcp_server_attributes(instance):
    return _get_toolset_attributes(instance)


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


def _get_tool_manager_module_path() -> str:
    try:
        import pydantic_ai.tool_manager  # noqa: F401

        return "pydantic_ai.tool_manager"
    except ImportError:
        return "pydantic_ai._tool_manager"


def _tool_manager_uses_execute_tool_call() -> bool:
    module_path = mlflow.pydantic_ai._get_tool_manager_module_path()
    try:
        module = __import__(module_path, fromlist=["ToolManager"])
        return hasattr(module.ToolManager, "execute_tool_call")
    except ImportError:
        return False


def _has_instrumentation_capability() -> bool:
    try:
        import pydantic_ai.capabilities.instrumentation  # noqa: F401

        return True
    except ImportError:
        return False


def setup_autologging() -> None:
    """Install the Pydantic AI 1.x autologging patches."""
    agent_methods = ["run", "run_sync", "run_stream"]

    try:
        from pydantic_ai import Agent

        if hasattr(Agent, "run_stream_sync"):
            agent_methods.append("run_stream_sync")
    except ImportError:
        pass

    has_instrumentation_capability = mlflow.pydantic_ai._has_instrumentation_capability()
    tool_manager_path = f"{mlflow.pydantic_ai._get_tool_manager_module_path()}.ToolManager"
    class_map = {
        "pydantic_ai.Agent": agent_methods,
        tool_manager_path: ["execute_tool_call"]
        if mlflow.pydantic_ai._tool_manager_uses_execute_tool_call()
        else ["handle_call"],
        "pydantic_ai.mcp.MCPServer": ["call_tool", "list_tools"],
    }
    if not has_instrumentation_capability:
        class_map["pydantic_ai.models.instrumented.InstrumentedModel"] = [
            "request",
            "request_stream",
        ]

    try:
        from pydantic_ai import Tool

        if hasattr(Tool, "run"):
            class_map["pydantic_ai.Tool"] = ["run"]
    except ImportError:
        pass

    try:
        from pydantic_ai import Agent

        original_init = Agent.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            return patched_agent_init(original_init, self, *args, **kwargs)

        patch = _wrap_patch(Agent, "__init__", patched_init)
        _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)
    except (ImportError, AttributeError) as e:
        _logger.error("Error patching Agent.__init__: %s", e)

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

    if has_instrumentation_capability:
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
