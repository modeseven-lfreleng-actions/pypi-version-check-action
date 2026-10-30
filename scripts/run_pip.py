# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
# Annotations raise TypeError on old Pythons this must run on.
# pyright: reportTypeCommentUsage=false

"""Run pip as 'python3 -m pip' would, without the workspace on its import path.

'python3 -m pip' puts the working directory, the caller's workspace,
first on sys.path, so a pip/ package there would be imported in place
of the installed one. Run as a script, Python puts this file's
directory first instead, and the working directory stays where pip
expects it: relative paths in PIP_* settings and pip.conf still
resolve against the workspace, and the user site is still searched.

The steps start this script without PYTHONPATH: its entries would come
before the standard library, and Python imports sitecustomize from them
at start-up. The entries the steps keep (see trusted_pythonpath.py)
arrive in ACTION_PYTHONPATH instead, and go after the standard library
and before the site directories, so a pip installed with --target is
still found ahead of site-packages, as with PYTHONPATH.

pip itself must then resolve outside the workspace and not above it;
otherwise this script stops, as if pip were missing, before importing
anything of it.
"""

import importlib.util
import os
import runpy
import site
import sys

# Run as a script, this file's directory is first on sys.path.
import trusted_pythonpath  # pyright: ignore[reportImplicitRelativeImport]


def _site_dirs():
    # type: () -> list[str]
    """The site directories on sys.path, as 'site' added them."""
    # Old virtualenv releases replace site.py with one lacking this
    found = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    if site.ENABLE_USER_SITE:
        found.append(site.getusersitepackages())
    return [os.path.normcase(os.path.abspath(path)) for path in found]


def add_action_pythonpath(entries):
    # type: (list[str]) -> None
    """Put ``entries`` on sys.path ahead of the first site directory."""
    site_dirs = _site_dirs()
    index = len(sys.path)
    for position, path in enumerate(sys.path):
        if os.path.normcase(os.path.abspath(path)) in site_dirs:
            index = position
            break
    sys.path[index:index] = [entry for entry in entries if entry]


def pip_locations():
    # type: () -> list[str]
    """Where 'import pip' would load from, found without importing it."""
    spec = importlib.util.find_spec("pip")
    if spec is None:
        return []
    found = list(spec.submodule_search_locations or [])
    if spec.has_location and spec.origin:
        found.append(spec.origin)
    return found


if __name__ == "__main__":
    add_action_pythonpath(os.environ.get("ACTION_PYTHONPATH", "").split(os.pathsep))
    roots = [
        os.path.realpath(root)
        for root in (os.getcwd(), os.environ.get("GITHUB_WORKSPACE", ""))
        if root
    ]
    for location in pip_locations():
        if trusted_pythonpath.reaches_workspace(location, roots):
            print("pip resolves to the workspace: " + location, file=sys.stderr)
            sys.exit(1)
    # pip ends the process through SystemExit; the namespace is unused.
    _ = runpy.run_module("pip", run_name="__main__", alter_sys=True)
