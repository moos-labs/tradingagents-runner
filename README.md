# tradingagents-runner

Runs the [TradingAgents](https://github.com/TauricResearch/TradingAgents)
multi-agent stock-research framework in GitHub Actions, following the
one-repo-per-tool runner pattern used by the other Moos Labs tool repos.

## How it works

- The workflow (`.github/workflows/run.yml`) is **manual only**
  (`workflow_dispatch`). It installs `requirements.txt`, then executes
  whatever shell commands are stored in the **COMMANDS** repository
  variable, and uploads everything under `results/` as a build artifact.
- The "Run commands" step receives these organization secrets as
  environment variables, so normal (LLM) runs can use them without a key
  ever appearing in a command or a log: **GOOGLE_API_KEY**,
  **SAMBANOVA_API_KEY**, **GH_MODELS_TOKEN** and **COHERE_API_KEY**.

## Providers

`scripts/ta_run.py --provider` accepts:

| Provider | Key env | Route |
| --- | --- | --- |
| `google` | GOOGLE_API_KEY | TradingAgents native Google client |
| `openai` | OPENAI_API_KEY | TradingAgents native OpenAI client |
| `anthropic` | ANTHROPIC_API_KEY | TradingAgents native Anthropic client |
| `sambanova` | SAMBANOVA_API_KEY | Generic `openai_compatible` client, backend `https://api.sambanova.ai/v1` |
| `github-models` | GH_MODELS_TOKEN | Generic `openai_compatible` client, backend `https://models.github.ai/inference` |
| `cohere` | COHERE_API_KEY | Generic `openai_compatible` client, backend `https://api.cohere.ai/compatibility/v1` |

Provider status from the first real runs (2026-11-10): the stored
GOOGLE_API_KEY's project is denied the Gemini free tier (403). The
SambaNova key authenticates and its model list answers, but chat
completions return 402 ("a payment method is required") on that account.
GitHub Models' endpoint answers plain-text `OK` stubs instead of JSON
(the service was measured in retirement brownout in late 2026), so it is
unusable. Cohere authenticates, lists `command-r-08-2024` and runs real
completions - the graph progressed through several analysts - but its
strict validator rejects TradingAgents' tool schema (`look_back_days`
is sent without a JSON-schema `type`), failing the News Analyst with a
400. Next step for a green run: patch the tool schemas for the Cohere
route, or add a payment method to the SambaNova account. For the
OpenAI-compatible providers, the script also writes
`results/provider_models.json` listing the models the endpoint serves.

## Commands

Smoke test (no LLM calls, no key needed):

```sh
mkdir -p results
python scripts/ta_run.py --smoke 2>&1 | tee results/run.log
```

Real research run:

```sh
mkdir -p results
set -o pipefail; python scripts/ta_run.py --ticker POUL.CA --date 2026-11-09 --provider cohere --model command-r-08-2024 2>&1 | tee results/run.log
```

Ticker, date, provider and model are script arguments, so the same repo
serves any market. Outputs land in `results/`: `decision.txt`,
`final_state.json` and `report.md` (plus `smoke.json` for smoke runs and
`provider_models.json` for the OpenAI-compatible providers).
