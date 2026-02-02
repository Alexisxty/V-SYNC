from __future__ import annotations


def ensure_qwen3_omni_config_compat() -> None:
    """Patch missing config attributes for newer Qwen3 Omni configs on old transformers."""
    try:
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )
    except Exception:
        return

    if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window"):
        Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = True


def ensure_transformers_no_init_weights() -> None:
    """Patch missing no_init_weights/ContextManagers for older/bleeding-edge transformers."""
    try:
        from transformers import modeling_utils
    except Exception:
        return

    if hasattr(modeling_utils, "no_init_weights") and hasattr(modeling_utils, "ContextManagers"):
        return

    try:
        from contextlib import ExitStack, contextmanager

        @contextmanager
        def no_init_weights(_enable: bool = True):  # type: ignore[name-defined]
            """
            Context manager to globally disable weight initialization to speed up loading large models.
            Mirrors transformers.modeling_utils.no_init_weights behavior.
            """
            if not _enable:
                yield
                return

            init_flag = getattr(modeling_utils, "_init_weights", True)
            try:
                modeling_utils._init_weights = False  # type: ignore[attr-defined]
                yield
            finally:
                modeling_utils._init_weights = init_flag  # type: ignore[attr-defined]

        if not hasattr(modeling_utils, "no_init_weights"):
            modeling_utils.no_init_weights = no_init_weights  # type: ignore[attr-defined]

        if not hasattr(modeling_utils, "ContextManagers"):
            class ContextManagers:  # type: ignore[no-redef]
                """
                Wrapper for `contextlib.ExitStack` which enters a collection of context managers.
                Adaptation of `ContextManagers` in the `fastcore` library.
                """

                def __init__(self, context_managers):  # type: ignore[no-untyped-def]
                    self.context_managers = context_managers
                    self.stack = ExitStack()

                def __enter__(self):  # type: ignore[no-untyped-def]
                    for context_manager in self.context_managers:
                        self.stack.enter_context(context_manager)

                def __exit__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                    self.stack.__exit__(*args, **kwargs)

            modeling_utils.ContextManagers = ContextManagers  # type: ignore[attr-defined]
    except Exception:
        return


def ensure_all_tied_weights_keys() -> None:
    """Ensure torch.nn.Module has all_tied_weights_keys attribute for older transformers."""
    try:
        import torch
    except Exception:
        return

    if hasattr(torch.nn.Module, "all_tied_weights_keys"):
        return

    def _all_tied_weights_keys(self):  # type: ignore[no-untyped-def]
        cached = getattr(self, "_all_tied_weights_keys", None)
        if isinstance(cached, dict):
            return cached

        getter = getattr(self, "get_expanded_tied_weights_keys", None)
        if callable(getter):
            try:
                expanded = getter()
            except TypeError:
                expanded = getter(all_submodels=True)
            if isinstance(expanded, dict):
                return expanded

        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, dict):
            return tied
        if isinstance(tied, (list, tuple, set)):
            return {key: key for key in tied}
        return {}

    def _set_all_tied_weights_keys(self, value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            self._all_tied_weights_keys = value
        else:
            self._all_tied_weights_keys = {}

    torch.nn.Module.all_tied_weights_keys = property(  # type: ignore[attr-defined]
        _all_tied_weights_keys, _set_all_tied_weights_keys
    )
