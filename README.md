# tradingagents-runner

Runs the [TradingAgents](https://github.com/TauricResearch/TradingAgents)
multi-agent stock-research framework in GitHub Actions, following the
one-repo-per-tool runner pattern used by the other Moos Labs tool repos.

## How it works

- The workflow (`.github/workflows/run.yml`) is **manual only**
  (`workflow_dispatch`). It installs `requirements.txt`, then executes
  whatever shell commands are stored in the **COMMANDS** repository
  variable, and uploads everything under `results/` as a build artifact.
- The "Run commands" step receives the repository secret
  **GOOGLE_API_KEY** as an environment variable, so normal (LLM) runs can
  use it without the key ever appearing in a command or a log.

## Commands

Smoke test (no LLM calls, no key needed) - this is the current COMMANDS value:

```sh
mkdir -p results
python scripts/ta_run.py --smoke 2>&1 | tee results/run.log
```

Real research run (once the GOOGLE_API_KEY secret exists):

```sh
mkdir -p results
python scripts/ta_run.py --ticker NVDA --date 2026-11-10 --provider google --model gemini-2.5-flash 2>&1 | tee results/run.log
```

Ticker, date, provider and model are script arguments, so the same repo
serves any market. Outputs land in `results/`: `decision.txt`,
`final_state.json` and `report.md` (plus `smoke.json` for smoke runs).
