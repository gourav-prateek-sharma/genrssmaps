"""genrssmaps — lightweight package init.

Keep imports lazy and minimal so importing the package doesn't
fail in environments that don't have optional heavy deps.
"""

__version__ = "0.1.0"
__author__ = "Gourav Prateek Sharma"

# Public API names (populate lazily)
__all__ = [
    "get_submodule",
    "lazy_import",
    "version",
]

def version():
    """Return package version."""
    return __version__

def get_submodule(name: str):
    """
    Import a submodule on demand and return it.
    Example: m = genrssmaps.get_submodule('gen_rss_csv')
    """
    import importlib
    return importlib.import_module(f"genrssmaps.{name}")

def lazy_import(name: str):
    """
    Convenience wrapper to access a symbol lazily:
      foo = genrssmaps.lazy_import('gen_rss_csv.rss_write_csv')
    """
    module_name, _, attr = name.rpartition(".")
    mod = get_submodule(module_name)
    return getattr(mod, attr)
