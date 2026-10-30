# AGENTS.md: KernelSHAP Streamer Operational Protocol

## 1. Project Overview & Core Mission
**KernelSHAP Streamer** is a high-performance, asynchronous Python library designed for real-time streaming feature attribution using the KernelSHAP algorithm. 
**Mission:** Provide low-latency, memory-efficient SHAP value estimation for streaming data pipelines, ensuring strict adherence to O(n) complexity per sample and zero-copy data handling where possible.

## 2. Local PowerShell Commands
Agents must execute these commands from the project root. Ensure `venv` is active.

*   **Setup:** `python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt -r requirements-dev.txt`
*   **Test:** `pytest tests/ --cov=kernelshap_streamer --cov-report=term-missing --durations=10`
*   **Run:** `python -m kernelshap_streamer.cli --config ./configs/default.yaml`
*   **Lint:** `ruff check . && mypy kernelshap_streamer/ --strict && black --check .`
*   **Clean:** `Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force`

## 3. Engineering Invariants
*   **Zero-Placeholder Mandate:** No `TODO`, `FIXME`, or `pass` blocks allowed in production code. If a feature is incomplete, it must be explicitly raised as a `NotImplementedError` with a tracking issue reference.
*   **Typing Strictness:** Every function, method, and class attribute must be annotated. Use `from __future__ import annotations`. Disallow `Any` unless explicitly justified in a comment block.
*   **Line Endings:** Enforce `LF` (Line Feed) for all source files. Repository configuration (`.gitattributes`) must be respected.
*   **Memory Management:** All streaming buffers must use `collections.deque` with `maxlen` or pre-allocated `numpy` arrays. No unbounded list growth.
*   **Concurrency:** Use `asyncio` for I/O-bound tasks; offload heavy SHAP kernel computations to `concurrent.futures.ProcessPoolExecutor`.

## 4. Self-Healing & Error Recovery Protocol
When a test fails or a runtime error occurs, follow this recursive loop:

1.  **Traceback Analysis:** Extract the full stack trace. Identify if the failure is a `TypeCheckError` (MyPy), `AssertionError` (PyTest), or `RuntimeError` (Logic).
2.  **Isolation:** Create a minimal reproduction script (`repro_issue_id.py`) that isolates the specific function failing.
3.  **Root Cause Identification:**
    *   If `MyPy`: Fix the signature mismatch or update the `TypeAlias`.
    *   If `Logic`: Verify the SHAP kernel weighting logic against the mathematical definition in `docs/math_spec.md`.
4.  **Verification:** Run the reproduction script. If it passes, integrate the fix into the main codebase and add a regression test in `tests/regressions/`.
5.  **Commit:** Atomic commits with the format: `fix(component): [issue_id] description of fix`.

## 5. Definition of Done (DoD) Quality Gate
A task is considered "Done" only when:
1.  **Type Safety:** `mypy --strict` returns 0 errors.
2.  **Coverage:** Unit test coverage is ≥ 95% for the modified module.
3.  **Linting:** `ruff` reports zero violations.
4.  **Documentation:** Docstrings follow Google Style Guide; `README.md` and `ARCHITECTURE.md` are updated if the API surface changed.
5.  **Performance:** Execution time for the modified module does not exceed the baseline by > 5% (measured via `pytest-benchmark`).
6.  **Review:** All code must be reviewed by a secondary agent or human peer, verifying that no "placeholder" logic remains.