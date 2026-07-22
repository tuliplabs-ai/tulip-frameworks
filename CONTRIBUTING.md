# Contributing to tulip-frameworks

Thanks for helping govern more of the agent ecosystem. This repo follows the
same bar as the core SDK: typed, linted, and tested — `hatch run check` must be
clean before a PR.

## Development setup

```bash
git clone https://github.com/tuliplabs-ai/tulip-frameworks.git
cd tulip-frameworks
pip install hatch
hatch run sdk        # install the core SDK (TULIP_SDK_DIR overrides ../tulip-agents)
hatch run check      # lint + types + tests with the coverage gate
```

Useful scripts:

```bash
hatch run test       # pytest
hatch run test-cov   # pytest with the ≥95% coverage gate
hatch run lint       # ruff
```

## Adding a framework bridge

Each bridge lives in its own module (`tulip_frameworks/<framework>.py`) and
follows the same pattern as the existing ones:

- Import the framework **lazily**; if it's missing, raise with the exact extra
  to install (`pip install "tulip-frameworks[<extra>]"`).
- The gated tool must be a first-class citizen of the framework — same name,
  schema, and description as the tool it wraps.
- Policy decisions come from the core `admit()`; no policy logic in the bridge.
- Ship a runnable example under `examples/` and offline tests (no framework
  API calls, no credentials).
- Declare the framework as an optional extra in `pyproject.toml`; the base
  `import tulip_frameworks` must keep pulling in **no** framework package.

## Pull requests

- Conventional Commit titles (`feat:`, `fix:`, `docs:` …).
- Keep the one-way dependency: this package imports `tulip-agents`; core never
  imports this package.
- CI runs lint, types, and tests on every PR — green required.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for coordinated disclosure.
