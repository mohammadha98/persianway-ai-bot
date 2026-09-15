"""NumPy 2.0 legacy-alias compatibility shim.

NumPy 2.0 removed several long-deprecated aliases from the top-level namespace
(``np.float_``, ``np.unicode_``, ``np.NaN``, ...). Libraries that still
reference them *at import time* then abort the whole process, e.g. ChromaDB
< 0.5.0 declares::

    ImageDType = Union[np.uint, np.int_, np.float_]

which produced the following production failure (every gunicorn worker exited
with code 3 and nginx answered 502)::

    AttributeError: `np.float_` was removed in the NumPy 2.0 release.

Importing this module restores the removed names on the already-imported
``numpy`` module, mapping each one to the exact dtype/function NumPy 2 renamed
it to. Nothing is overridden when the name still exists, so this module is a
no-op on NumPy 1.x.

It is a safety net for third-party code we do not control (``requirements.txt``
already pins ChromaDB to a NumPy-2 compatible release). It must be imported
*before* the affected library, therefore it is loaded at the top of
``main.py`` and of ``app/services/document_processor.py``.
"""

import numpy as np

# Pre-NumPy-2 name -> NumPy 2 equivalent. Values are the exact replacements
# documented by NumPy (``np.float_`` -> ``np.float64`` etc.).
_LEGACY_ALIASES = {
    "float_": np.float64,
    "singlecomplex": np.complex64,
    "cfloat": np.complex128,
    "complex_": np.complex128,
    "longfloat": np.longdouble,
    "clongfloat": np.clongdouble,
    "longcomplex": np.clongdouble,
    "unicode_": np.str_,
    "string_": np.bytes_,
    "NaN": np.nan,
    "Inf": np.inf,
    "Infinity": np.inf,
    "infty": np.inf,
    "NINF": -np.inf,
    "PINF": np.inf,
    "round_": np.round,
    "product": np.prod,
    "cumproduct": np.cumprod,
    "sometrue": np.any,
    "alltrue": np.all,
    "bool8": np.bool_,
}


def install_legacy_numpy_aliases() -> list:
    """Restore the aliases removed in NumPy 2.0, if they are missing.

    Returns:
        The names that were actually added (empty on NumPy 1.x).
    """
    restored = []
    for name, value in _LEGACY_ALIASES.items():
        if not hasattr(np, name):
            try:
                setattr(np, name, value)
                restored.append(name)
            except Exception:  # pragma: no cover - defensive only
                pass
    return restored


# Applied at import time: `import app.core.numpy_compat` is enough to make
# legacy third-party imports work again.
RESTORED_ALIASES = install_legacy_numpy_aliases()
