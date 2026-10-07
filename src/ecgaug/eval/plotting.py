"""Plotting helpers. Import pyplot through :func:`import_pyplot`, never at module import time."""

from __future__ import annotations


def import_pyplot():
    """Return ``matplotlib.pyplot`` on the non-interactive Agg backend.

    On some macOS versions ``system_profiler SPFontsDataType`` returns no font list, and matplotlib's
    first font scan then fails with ``KeyError('_items')``. The scan is retried with that step
    skipped, so matplotlib uses the standard font directories and writes its cache; later imports
    (including captum's) then work unpatched.
    """
    import matplotlib

    matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except KeyError as err:
        if err.args != ("_items",):
            raise
        import subprocess

        original = subprocess.check_output

        def without_font_listing(cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "system_profiler":
                raise OSError("system_profiler font listing unavailable")
            return original(cmd, *args, **kwargs)

        subprocess.check_output = without_font_listing
        try:
            import matplotlib.pyplot as plt
        finally:
            subprocess.check_output = original
    return plt
