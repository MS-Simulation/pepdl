"""P0 gate: torch is an *optional* extra (`pepdl[local]`).

These tests assert the torch-decoupling contract:
  1. `import pepdl` works with no torch installed.
  2. The Koina (remote) path and its wrappers construct with no torch.
  3. Any *local* model routes a missing torch through `require_torch`, which raises an actionable
     `pip install 'pepdl[local]'` hint — never a bare `ModuleNotFoundError: No module named 'torch'`
     and never a silent `None`.

This is the gate that pepdl was missing. tests/test_import_gate.py used to stand in for it by
listing `torch` as a forbidden import, but that asserts the opposite kind of thing — that torch is
never LOADED when it *is* installed — which pepdl never promised and does not do. The claim worth
protecting is that torch can be ABSENT, and you cannot check that by looking at `sys.modules` in an
environment where it is present. Hence `block_torch()`.

To stay runnable in a torch-*installed* dev env, `block_torch()` makes `import torch` fail
in-process (meta-path finder + sys.modules eviction), so a lazy `import torch` at point of use
behaves exactly as it would where torch was never installed.

Ported from imspy-predictors' gate of the same name, which pepdl was extracted from; the two are
kept in step deliberately.
"""

import sys

import pytest


class block_torch:
    """Context manager: make ``import torch`` raise ImportError in-process.

    Evicts already-imported ``torch`` submodules and installs a meta-path
    finder that refuses them, so lazy ``import torch`` at point-of-use behaves
    exactly as it would in an env where torch was never installed.
    """

    def __init__(self):
        self._saved = {}

    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named '{name}' (blocked by block_torch)")
        return None

    def __enter__(self):
        for mod in list(sys.modules):
            if mod == "torch" or mod.startswith("torch."):
                self._saved[mod] = sys.modules.pop(mod)
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        # Ordered so a failure to unhook cannot strand the process without torch: removing the
        # finder is best-effort, restoring sys.modules is not. Bare `remove` would raise
        # ValueError if the block reordered sys.meta_path, skipping the restore below and
        # masking whatever exception was already propagating.
        try:
            if self in sys.meta_path:
                sys.meta_path.remove(self)
        finally:
            # Purge anything the block let in under the same names before putting the real
            # modules back, so `update` cannot leave a half-blocked torch behind.
            for mod in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
                del sys.modules[mod]
            sys.modules.update(self._saved)
            self._saved.clear()
        return False


class fresh_import:
    """Force a re-import of ``prefix`` inside the block, then put the originals back.

    ``block_torch`` alone is not enough for anything that inspects state decided at import time
    (the ``TORCH_AVAILABLE`` branches): those modules are already in ``sys.modules``, imported
    *with* torch, so they must be re-imported under the block to see the torch-free branch.

    The restore half matters as much as the evict half. Dropping the package and walking away
    leaks into every later test in the process: the re-import builds NEW class objects, so
    `isinstance` checks elsewhere compare against classes that are equal by name and distinct by
    identity, and fail. That was real, in the sibling this gate is ported from — running it before
    imspy-predictors' tests/test_utility.py turned 1 genuine failure into 7, the 6 extras being
    pure cross-test pollution. pepdl has no test that trips it today; the restore keeps it so.

    Scope, deliberately narrow: TOP-LEVEL package names only. A dotted prefix would leave the
    retained parent package holding its old attribute — `fresh_import("pepdl.ccs")`
    rebinds `pepdl.ccs` to the temporary module, and restoring `sys.modules` does not
    put the parent's attribute back, so the module graph splits. Rather than half-support that,
    it raises. What this does NOT undo either way: references the block handed to modules outside
    the prefix, and import side effects on anything else.
    """

    def __init__(self, prefix):
        if "." in prefix:
            raise ValueError(
                f"fresh_import is top-level only, got {prefix!r}: restoring sys.modules would "
                "leave the parent package's attribute pointing at the temporary module."
            )
        self.prefix = prefix
        self._saved = {}

    def _matching(self):
        return [m for m in list(sys.modules)
                if m == self.prefix or m.startswith(self.prefix + ".")]

    def __enter__(self):
        try:
            for mod in self._matching():
                self._saved[mod] = sys.modules.pop(mod)
        except BaseException:
            self.__exit__()  # a partial eviction is exactly the leak this class exists to prevent
            raise
        return self

    def __exit__(self, *exc):
        try:
            for mod in self._matching():  # whatever the block imported
                del sys.modules[mod]
        finally:
            sys.modules.update(self._saved)
            self._saved.clear()
        return False


def test_package_imports_without_torch():
    """`import pepdl` must succeed with torch unavailable."""
    with block_torch():
        with fresh_import("pepdl"):
            import pepdl  # noqa: F401

            assert pepdl is not None


def test_require_torch_raises_actionable_hint():
    """The helper names the fix and reassures that Koina needs no torch."""
    from pepdl.utility import require_torch

    with block_torch():
        with pytest.raises(ImportError) as ei:
            require_torch("some local model")

    msg = str(ei.value)
    assert "pepdl[local]" in msg
    assert "Koina" in msg
    # never leak the raw failure as the primary message
    assert "some local model" in msg


@pytest.mark.parametrize(
    "module, cls",
    [
        ("pepdl.ccs.predictors", "DeepPeptideIonMobilityApex"),
        ("pepdl.rt.predictors", "DeepChromatographyApex"),
        ("pepdl.intensity.predictors", "DeepPeptideIntensityPredictor"),
    ],
)
def test_local_predictor_instantiation_raises_hint(module, cls):
    """Instantiating a local (torch) predictor without torch → the hint."""
    import importlib

    with block_torch():
        mod = importlib.import_module(module)
        predictor_cls = getattr(mod, cls)
        with pytest.raises(ImportError) as ei:
            predictor_cls()

    assert "pepdl[local]" in str(ei.value)


def test_koina_wrapper_constructs_without_torch():
    """The Prosit/Koina intensity wrapper is torch-free to construct."""
    with block_torch():
        from pepdl.intensity.predictors import Prosit2023TimsTofWrapper

        wrapper = Prosit2023TimsTofWrapper()  # default use_koina=True; no torch touched
        assert wrapper.use_koina is True


def test_torch_only_alias_is_not_a_silent_none():
    """`SquareRootProjectionLayer` is re-exported from `pepdl.ccs`, so without torch it
    used to be an importable `None` that failed as `TypeError: 'NoneType' object is not callable` —
    naming neither torch nor the fix. It must fail through `require_torch` like every other local
    path."""
    with block_torch():
        with fresh_import("pepdl"):
            from pepdl.ccs import SquareRootProjectionLayer

            assert SquareRootProjectionLayer is not None, "silent None: the contract forbids this"
            with pytest.raises(ImportError) as ei:
                SquareRootProjectionLayer()

    assert "pepdl[local]" in str(ei.value)
