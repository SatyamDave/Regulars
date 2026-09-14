# Security scanning

Regulars is scanned for known vulnerabilities on two separate surfaces: the container
image (OS packages + base image) and the Python dependency tree (`uv.lock`). These are
different attack surfaces with different fix paths, so they run as separate CI jobs.

## What's scanned, and when

### Container image — `container-scan` job in `.github/workflows/docker.yml`

- Runs on every push to `main`, every `v*` tag, and every pull request that touches
  `Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `regulars/**`, or the
  workflow file itself — the same trigger as the existing `image` build job.
- Builds the image locally with `docker/build-push-action` (`push: false`, `load: true`,
  tagged `regulars:scan`) using the same `Dockerfile` and build cache as the real build,
  so what's scanned matches what would ship.
- Scans the built image with [`aquasecurity/trivy-action`](https://github.com/aquasecurity/trivy-action)
  (pinned to `v0.32.0`), covering OS packages (Debian packages in the `python:3.12-slim-bookworm`
  runtime layer) and Python packages baked into the image's virtualenv.
- Fails the job (`exit-code: "1"`) on any `CRITICAL` or `HIGH` severity finding that has a
  known fix available (`ignore-unfixed: true` — findings with no upstream fix yet are
  reported but don't block CI, since there is nothing actionable to do about them today).
- Uploads results in SARIF format via `github/codeql-action/upload-sarif`, unconditionally
  (`if: always()`), so findings appear in the repo's Security tab even when the scan step
  fails the job.
- Runs independently of the `image` job — it does not gate the build/push step, so a
  scan failure surfaces as a red check without blocking the existing `image` job's
  behavior (build-push-action's `push` is still driven only by `event_name`, unchanged).

### Python dependencies — `dependency-audit` job in `.github/workflows/ci.yml`

- Runs on the same triggers as the rest of CI (push to `main`, pull requests, merge queue).
- Exports the locked, non-dev, all-extras dependency set from `uv.lock` to a plain
  `requirements.txt` (`uv export --frozen --no-dev --all-extras --no-emit-project`) and
  runs `uvx pip-audit -r requirements-audit.txt --desc` against it via the PyPI/OSV
  advisory databases.
- `--no-emit-project` excludes the local editable `regulars` package itself (which has no
  PyPI advisories and can't be hash-installed as an editable requirement), leaving only
  third-party dependencies.
- This catches CVEs in Python packages regardless of whether they end up in the container
  image (e.g. a `dev`-only tool would not — but everything in `--no-dev --all-extras` does
  reach the image).
- Wired into the existing `ci-passed` rollup job's `needs` list, so a CVE finding here
  blocks the same required status check as lint/typecheck/test failures do.

## How to interpret Security-tab findings

Container scan findings land under **Security → Code scanning alerts**, tagged with the
tool name `Trivy`. Each alert shows:

- The CVE ID and a link to its advisory.
- The affected package and installed version, and the first version containing a fix.
- Severity (only `CRITICAL`/`HIGH` reach this tab, per the `severity` filter above).
- Which layer introduced it — base OS package vs. a Python package installed by `uv sync`.

For an OS-package finding, the fix is almost always bumping the base image tag in the
`Dockerfile` (`python:3.12-slim-bookworm` / `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`)
to pick up the upstream Debian security patch, then rebuilding.

For a Python-package finding, the fix is bumping the affected dependency's version
constraint in `pyproject.toml` and re-running `uv lock`, then confirming the version bump
doesn't break the test suite.

Dependency-audit (`pip-audit`) findings show up directly in the failed `dependency-audit`
job's log, with the CVE ID, affected package/version, and the fixed version — they are not
mirrored to the Security tab (pip-audit doesn't emit SARIF in this setup; only the
container scan does).

## Accepting a finding as a false positive or accepted risk

Trivy findings can be suppressed with a `.trivyignore` file at the repo root, one
finding ID per line, with a comment explaining why:

```
# .trivyignore
# CVE-2023-XXXXX: affects libX only when built with feature Y, which this image's
# base doesn't enable. Upstream fix not yet released for bookworm. Re-check monthly.
CVE-2023-XXXXX
```

Point Trivy at it by adding `trivyignores: .trivyignore` to the `with:` block of the
`Scan image with Trivy` step in `.github/workflows/docker.yml`. Only add an entry when:

- the finding is genuinely not exploitable in how the image runs it (e.g. a vulnerable
  code path that Regulars never calls), or
- there is no fix available yet and the risk has been explicitly accepted,

and always with a comment stating which of those applies and a re-check date. Do not
suppress a finding just because it's inconvenient — prefer bumping the version.

As of this setup, no `.trivyignore` exists: the current image and dependency tree have no
known findings to suppress (see below).

## Current scan status (as of this change)

Docker was not available in the environment where this scanning setup was added, so the
container image itself has not yet been scanned by this pipeline — that will happen on
the first push/PR that triggers `container-scan`. The Python dependency surface *was*
scanned directly, as a substitute check:

```
uv export --frozen --no-dev --all-extras --no-emit-project --format requirements-txt -o requirements-audit.txt
uvx pip-audit -r requirements-audit.txt --desc
```

Result: **no known vulnerabilities found** in the current locked dependency set
(`uv.lock` as of this commit). No dependency version changes were made as part of this
change — there was nothing to fix.
