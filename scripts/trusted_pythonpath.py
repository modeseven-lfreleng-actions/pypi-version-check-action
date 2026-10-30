# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
# Annotations raise TypeError on old Pythons this must run on; see below.
# pyright: reportTypeCommentUsage=false

"""Keep the workspace off the import path of the Python that runs pip.

The working directory is the caller's workspace. Settings a caller
passes in can still let the checkout supply code that pip's
interpreter runs:

- PYTHONPATH: its entries come before the standard library, and
  Python imports sitecustomize from them at start-up; an entry that is
  empty or relative (both resolved against the working directory),
  that resolves inside the workspace, or that is a parent of it, so
  the checkout imports as a package named after its directory, puts
  the checkout there, and so does a module in an entry elsewhere that
  links into it;
- a user site directory inside or above the workspace
  (PYTHONUSERBASE, or HOME when that is unset); Python runs the .pth
  files it finds there at every start-up, and adds it to sys.path;
- PYTHONHOME whose standard-library directories resolve into the
  workspace, which loads the standard library from there,
  PYTHONPYCACHEPREFIX there, whose cached bytecode would stand in for
  modules' own, and a PYTHONPLATLIBDIR that is a path rather than a
  directory name such as lib64.

(On macOS, PYTHONEXECUTABLE and __PYVENV_LAUNCHER__ also tell Python
where it lives. It reads them even under -I, so the steps drop them
before any Python, this script included, starts.)

Run with -I -S, so that it ignores PYTHONPATH, PYTHONHOME and the
working directory and runs no .pth file, this script prints shell code
for 'eval'. PYTHONPATH is always unset; its absolute entries outside
the workspace and not above it, such as a 'pip install --target'
directory, go to ACTION_PYTHONPATH, which run_pip.py adds to sys.path
after the standard library. A user site inside or above the workspace
is replaced: PYTHONUSERBASE moves to FALLBACK_USER_BASE, an absolute
directory outside the workspace that the steps share, so pip can still
fall back to a user install there; without one, PYTHONNOUSERSITE=1
turns the user site off. A user site elsewhere is left alone.
PYTHONHOME is unset when any part of it, or any standard-library
directory under it, is relative or resolves inside or above the
workspace; PYTHONPYCACHEPREFIX when any part of it does; and
PYTHONPLATLIBDIR when it holds a path separator or '..'.

With --check-site, run with -S once those settings apply, it exits 1
with a message when a site-packages directory of the interpreter
itself, or a path a .pth file there adds, lies inside or above the
workspace: an environment outside it that links its site-packages
there, or an editable install of the checkout. Without -I, the
settings place the interpreter as they will for pip; -S runs no .pth
file, and the .pth files are only read.

It runs on any Python 3 a runner may have, so its types are given as
comments, which no version of Python evaluates.
"""

import os
import shlex
import site
import sys


def _inside(path, roots):
    # type: (str, list[str]) -> bool
    """True when ``path`` resolves to one of ``roots`` or below it."""
    real = os.path.realpath(path)
    return any(
        real == root or real.startswith(root.rstrip(os.sep) + os.sep) for root in roots
    )


def _above(path, roots):
    # type: (str, list[str]) -> bool
    """True when ``path`` resolves to a directory holding one of ``roots``."""
    real = os.path.realpath(path).rstrip(os.sep) + os.sep
    return any(root.startswith(real) for root in roots)


def _reaches(path, roots):
    # type: (str, list[str]) -> bool
    """True when ``path`` is relative, or resolves inside or above a root."""
    return not os.path.isabs(path) or _inside(path, roots) or _above(path, roots)


def reaches_workspace(path, real_roots):
    # type: (str, list[str]) -> bool
    """True when ``path`` resolves inside or above a root (already resolved)."""
    return _inside(path, real_roots) or _above(path, real_roots)


def trusted_pythonpath(pythonpath, roots):
    # type: (str, list[str]) -> str
    """Keep the absolute entries of ``pythonpath`` neither in nor above a root."""
    kept = [
        entry for entry in pythonpath.split(os.pathsep) if not _reaches(entry, roots)
    ]
    return os.pathsep.join(kept)


def _home_dirs(home):
    # type: (str) -> list[str]
    """Each part of ``home`` and the standard-library directories under it."""
    major, minor = sys.version_info[:2]
    version = f"python{major}.{minor}"
    libdirs = ["lib", "lib64", getattr(sys, "platlibdir", "lib"), "Lib", "DLLs"]
    found = []  # type: list[str]
    for part in home.split(os.pathsep):
        found.append(part)
        if not os.path.isabs(part):
            continue
        for libdir in libdirs:
            found.append(os.path.join(part, libdir))
            for name in (version, version + "t"):
                found.append(os.path.join(part, libdir, name))
                found.append(os.path.join(part, libdir, name, "lib-dynload"))
        found.append(os.path.join(part, "lib", f"python{major}{minor}.zip"))
    return found


def shell_settings(pythonpath, user_site, roots, fallback_user_base=""):
    # type: (str, str, list[str], str) -> str
    """Shell code that applies the settings described in the module docstring."""
    real_roots = [os.path.realpath(root) for root in roots if root]
    kept = trusted_pythonpath(pythonpath, real_roots)
    lines = [
        "unset PYTHONPATH",
        "export ACTION_PYTHONPATH=" + shlex.quote(kept)
        if kept
        else "unset ACTION_PYTHONPATH",
    ]
    if user_site and (_inside(user_site, real_roots) or _above(user_site, real_roots)):
        if not _reaches(fallback_user_base, real_roots):
            lines.append("export PYTHONUSERBASE=" + shlex.quote(fallback_user_base))
        else:
            lines.append("export PYTHONNOUSERSITE=1")
    home = os.environ.get("PYTHONHOME", "")
    if home and any(_reaches(path, real_roots) for path in _home_dirs(home)):
        lines.append("unset PYTHONHOME")
    cache_prefix = os.environ.get("PYTHONPYCACHEPREFIX", "")
    if cache_prefix and any(
        _reaches(part, real_roots) for part in cache_prefix.split(os.pathsep)
    ):
        lines.append("unset PYTHONPYCACHEPREFIX")
    platlibdir = os.environ.get("PYTHONPLATLIBDIR", "")
    if os.sep in platlibdir or ".." in platlibdir:
        lines.append("unset PYTHONPLATLIBDIR")
    return "\n".join(lines)


def _venv_prefix(executable):
    # type: (str) -> str
    """The prefix site.venv() gives ``executable``, or '' outside a venv."""
    exe_dir = os.path.dirname(os.path.abspath(executable))
    site_prefix = os.path.dirname(exe_dir)
    for directory in (exe_dir, site_prefix):
        if os.path.isfile(os.path.join(directory, "pyvenv.cfg")):
            return site_prefix
    return ""


def own_site_dirs():
    # type: () -> list[str]
    """Every site-packages directory 'site' may add for this interpreter.

    A superset: those of the current and base prefixes, and of the
    virtual environment site.venv() finds from the executable, whether
    or not that environment includes the base one. Python 3.14 sets the
    prefix of a virtual environment even under -S; earlier versions
    leave that to 'site'.
    """
    prefixes = [
        sys.prefix,
        sys.exec_prefix,
        sys.base_prefix,
        sys.base_exec_prefix,
        _venv_prefix(sys.executable),
    ]
    return site.getsitepackages([prefix for prefix in prefixes if prefix])


def _pth_paths(directory):
    # type: (str) -> list[str]
    """The paths the .pth files in ``directory`` add, read without running them.

    As site.addpackage() reads them: blank lines and comments are
    skipped, lines starting with 'import' are code, and any other line
    names a path relative to ``directory``.
    """
    found = []  # type: list[str]
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return found
    for name in names:
        if not name.endswith(".pth") or name.startswith("."):
            continue
        try:
            with open(os.path.join(directory, name), "rb") as handle:
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                continue
            found.append(os.path.join(directory, line.rstrip()))
    return found


def site_dirs_reaching(roots):
    # type: (list[str]) -> list[str]
    """Site-packages directories, or paths their .pth files add, in or above a root.

    The user site counts too, when it is on; the filter has already
    moved one that is itself in or above the workspace.
    """
    real_roots = [os.path.realpath(root) for root in roots if root]
    directories = own_site_dirs()
    if site.check_enableusersite():
        directories.append(site.getusersitepackages())
    reaching = []  # type: list[str]
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for path in [directory] + _pth_paths(directory):
            if reaches_workspace(path, real_roots):
                reaching.append(path)
    return reaching


if __name__ == "__main__":
    workspace_roots = [os.getcwd(), os.environ.get("GITHUB_WORKSPACE", "")]
    if sys.argv[1:] == ["--check-site"]:
        reaching = site_dirs_reaching(workspace_roots)
        for found in reaching:
            print(
                sys.executable + ": imports from the workspace: " + found,
                file=sys.stderr,
            )
        sys.exit(1 if reaching else 0)
    print(
        shell_settings(
            os.environ.get("PYTHONPATH", ""),
            site.getusersitepackages(),
            workspace_roots,
            os.environ.get("FALLBACK_USER_BASE", ""),
        )
    )
