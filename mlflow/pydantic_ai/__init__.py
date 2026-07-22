import importlib.metadata

from packaging.version import Version

from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "pydantic_ai"


def _is_pydantic_ai_v2() -> bool:
    """Return whether the installed Pydantic AI release uses the 2.x API."""
    try:
        return Version(importlib.metadata.version("pydantic-ai")).major >= 2
    except importlib.metadata.PackageNotFoundError:
        return False


# These private helpers remain available for the Pydantic AI 1.x tests and for
# compatibility with code that imported them while the integration was unsplit.
def _get_tool_manager_module_path() -> str:
    from mlflow.pydantic_ai.autolog_v1 import _get_tool_manager_module_path

    return _get_tool_manager_module_path()


def _tool_manager_uses_execute_tool_call() -> bool:
    from mlflow.pydantic_ai.autolog_v1 import _tool_manager_uses_execute_tool_call

    return _tool_manager_uses_execute_tool_call()


def _has_instrumentation_capability() -> bool:
    from mlflow.pydantic_ai.autolog_v1 import _has_instrumentation_capability

    return _has_instrumentation_capability()


@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enable (or disable) autologging for Pydantic_AI.

    Args:
        log_traces: If True, capture spans for agent + model calls.
        disable:   If True, disable the autologging patches.
        silent:    If True, suppress MLflow warnings/info.
    """
    if _is_pydantic_ai_v2():
        from mlflow.pydantic_ai.autolog_v2 import setup_autologging
    else:
        from mlflow.pydantic_ai.autolog_v1 import setup_autologging

    setup_autologging()
    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
