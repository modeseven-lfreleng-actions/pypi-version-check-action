<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2024 The Linux Foundation
-->

# 📦 PyPI Package Check

Checks the Python package index for the presence of a package (and
optionally a specific release/version).

## pypi-version-check-action

## Usage Example

Pass the index query URL, along with Python package name and optionally the
package release/version to check.

```yaml
  - name: "Checking package index for build/release"
    id: check-package-index
    uses: lfreleng-actions/pypi-version-check-action@main
    with:
        index_url: "https://pypi.org/simple"
        package_name: "ITR"
        package_version: "1.1.10"
```

Query URL examples:

- PyPI <https://pypi.org/simple>
- TestPyPI <https://test.pypi.org/simple>

## Inputs

<!-- markdownlint-disable MD013 -->

| Variable Name   | Required | Default                   | Description                                  |
| --------------- | -------- | ------------------------- | -------------------------------------------- |
| PACKAGE_NAME    | True     |                           | Name of Python package to check              |
| PACKAGE_VERSION | False    |                           | Optional release/version to check            |
| INDEX_URL       | False    | <https://pypi.org/simple> | The Python package index server URL to query |
| PRE_RELEASE     | False    | False                     | Also checks pre-release/development versions |
| EXIT_ON_FAIL    | False    | False                     | Set true to exit with error on failure       |

<!-- markdownlint-enable MD013 -->

Note: Leading "v" characters are automatically stripped, as the Python package
index uses purely numerical versioning.

## Outputs

<!-- markdownlint-disable MD013 -->

| Variable Name | Mandatory | Description                                    |
| ------------- | --------- | ---------------------------------------------- |
| PACKAGE_MATCH | True      | Always reports the presence of the project     |
| VERSION_MATCH | False     | Reports a match when specific version provided |

<!-- markdownlint-enable MD013 -->

## Implementation Details

This action uses the "pip index" command, which is an experimental command.
It may get removed/changed in a future release without prior warning.

The action runs pip with the first "python3" on the PATH that lies outside
the caller's workspace, and stops with an error when it finds none, or when
that interpreter's site-packages are in the workspace.
When that interpreter lacks pip, the action installs it with "ensurepip",
and fails with an error if that is not possible.

To query development and pre-release packages, the action implementation uses
the "--pre" command-line flag to gather the required versioning information.

## Development

The tests under `tests/` run with [uv](https://docs.astral.sh/uv/), which
installs the versions pinned in `uv.lock` into a local `.venv`:

```bash
uv sync --frozen
uv run --frozen pytest
```
