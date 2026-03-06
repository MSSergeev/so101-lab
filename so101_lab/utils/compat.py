# Runs in: lerobot-env (Python 3.12)
"""Compatibility patches for third-party libraries."""


def patch_hf_custom_models() -> None:
    """Fix transformers >=5.3 breaking custom HF models that lack all_tied_weights_keys.

    transformers 5.3.0 added PreTrainedModel.all_tied_weights_keys (set in __init__),
    but custom models loaded via trust_remote_code=True may not call super().__init__
    in the expected order, leaving the attribute missing.

    Patches mark_tied_weights_as_initialized to inject the attribute before it's accessed.
    Call this once before any SACPolicy creation.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        return

    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, loading_info)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched
