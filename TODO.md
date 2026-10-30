# TODO.md: KernelSHAP Streamer Roadmap

## Phase 1: Architecture Scaffolding & Foundation
- [ ] Initialize repository structure with `src/`, `tests/`, `docs/`, and `benchmarks/` directories.
- [ ] Configure `pyproject.toml` with Poetry, enforcing Python 3.11+ and strict `mypy` settings.
- [ ] Implement base abstract classes for `KernelSHAPEngine` and `DataStreamBuffer`.
- [ ] Set up CI/CD pipeline (GitHub Actions) for linting (`ruff`), type checking (`mypy`), and unit testing (`pytest`).
- [ ] Define core data structures (Pydantic models) for feature vectors and SHAP value outputs.

**Definition of Done (DoD):**
- Project builds cleanly in a clean-room environment.
- 100% type coverage verified by `mypy --strict`.
- CI pipeline passes on initial commit.

## Phase 2: Core Domain Logic & Engine Implementation
- [ ] Implement the `KernelSHAP` perturbation logic using optimized NumPy vectorization.
- [ ] Develop the `StreamBuffer` with sliding-window semantics for high-throughput data ingestion.
- [ ] Integrate `scikit-learn` compatible model wrappers for black-box prediction access.
- [ ] Implement the weighted least squares (WLS) solver for SHAP coefficient estimation.
- [ ] Add support for asynchronous stream processing using `asyncio` and `multiprocessing` for CPU-bound kernel calculations.

**Definition of Done (DoD):**
- KernelSHAP engine produces mathematically consistent values against static reference implementations.
- Memory footprint remains constant under sustained stream pressure.
- Asynchronous throughput exceeds 10k perturbations/sec on standard hardware.

## Phase 3: Automated Stress Testing & Verification (DoD Gates)
- [ ] Develop property-based tests using `Hypothesis` to verify SHAP properties (Efficiency, Symmetry, Dummy, Additivity).
- [ ] Create load-testing suite to measure latency distribution (p99) under varying feature dimensions.
- [ ] Implement regression testing against known model outputs (e.g., XGBoost/LightGBM benchmarks).
- [ ] Establish performance benchmarks for kernel matrix inversion stability.

**Definition of Done (DoD):**
- All SHAP axioms verified via property-based testing.
- P99 latency documented and verified under 50ms for 10-feature vectors.
- Zero memory leaks detected via `valgrind` or `tracemalloc` during 1-hour soak test.

## Phase 4: Packaging, Documentation & README
- [ ] Generate API documentation using `mkdocs` and `mkdocstrings`.
- [ ] Write comprehensive usage examples for real-time inference scenarios (e.g., Kafka/Redpanda integration).
- [ ] Finalize `README.md` with architectural diagrams, installation instructions, and performance metrics.
- [ ] Configure `pyproject.toml` for PyPI distribution (metadata, classifiers, dependencies).

**Definition of Done (DoD):**
- Documentation is auto-generated and reflects current source code.
- README includes a "Getting Started" guide that is verified to work in a fresh virtual environment.
- All public APIs are documented with docstrings following Google style.

## Phase 5: Release & Production Readiness
- [ ] Conduct final security audit (dependency scanning via `safety` and `bandit`).
- [ ] Tag v0.1.0 release and publish to PyPI.
- [ ] Create a `CONTRIBUTING.md` file outlining PR standards and local development workflow.
- [ ] Establish a `SECURITY.md` file with vulnerability disclosure protocols.

**Definition of Done (DoD):**
- Zero high-severity vulnerabilities identified by security tooling.
- Release artifacts signed and verified.
- Repository is fully compliant with open-source production standards.