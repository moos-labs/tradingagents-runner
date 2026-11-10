#!/usr/bin/env python3
"""Run TradingAgents for a ticker/date, or a no-LLM smoke test.

Generic by design: ticker, analysis date, LLM provider and model are all
command-line arguments, so the same runner serves any market.

Smoke mode (--smoke) imports the package, prints its version and confirms
the TradingAgentsGraph class imports cleanly. It makes NO LLM calls and
needs NO API key.

Normal mode reads the provider API key from the environment, runs the
multi-agent graph for the ticker/date and saves the decision plus report
sections under results/.

Providers:
  google        GOOGLE_API_KEY      (native Google client)
  openai        OPENAI_API_KEY      (native OpenAI client)
  anthropic     ANTHROPIC_API_KEY   (native Anthropic client)
  sambanova     SAMBANOVA_API_KEY   (TradingAgents "openai_compatible"
                                     client pointed at api.sambanova.ai/v1)
  github-models GH_MODELS_TOKEN     (same generic client pointed at
                                     models.github.ai/inference)
"""
import argparse
import json
import os
import sys
from datetime import date as date_cls
from pathlib import Path

RESULTS = Path("results")

# Providers routed through TradingAgents v0.7.1's generic
# "openai_compatible" client: the framework points it at the config's
# backend_url and reads the key from OPENAI_COMPATIBLE_API_KEY, so the
# provider's own key env is copied into that name in-process (never
# printed, never logged).
COMPAT_PROVIDERS = {
    "sambanova": {
        "backend_url": "https://api.sambanova.ai/v1",
        "key_env": "SAMBANOVA_API_KEY",
    },
    "github-models": {
        "backend_url": "https://models.github.ai/inference",
        "key_env": "GH_MODELS_TOKEN",
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TradingAgents runner")
    p.add_argument("--ticker", default=None,
                   help="Ticker symbol, e.g. NVDA or POUL.CA")
    p.add_argument("--date", default=None,
                   help="Analysis date YYYY-MM-DD (default: today, UTC)")
    p.add_argument("--provider", default="google",
                   help="LLM provider: google, openai, anthropic, "
                        "sambanova or github-models (default: google)")
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Model for both deep and quick thinking (default: gemini-2.5-flash)")
    p.add_argument("--smoke", action="store_true",
                   help="Import check only; no LLM calls, no key needed")
    return p.parse_args(argv)


def smoke():
    RESULTS.mkdir(parents=True, exist_ok=True)
    info = {"ok": False, "mode": "smoke"}
    try:
        import importlib.metadata as md
        try:
            info["tradingagents_version"] = md.version("tradingagents")
        except Exception as e:  # noqa: BLE001
            info["version_error"] = f"{type(e).__name__}: {e}"
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        info["graph_class"] = (
            f"{TradingAgentsGraph.__module__}.{TradingAgentsGraph.__name__}"
        )
        from tradingagents.default_config import DEFAULT_CONFIG
        info["default_config_keys"] = sorted(DEFAULT_CONFIG.keys())
        info["ok"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    (RESULTS / "smoke.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("SMOKE OK" if info["ok"] else "SMOKE FAILED")
    return 0 if info["ok"] else 1


def probe_models(backend_url, api_key, model):
    """Best-effort list of the models the endpoint serves.

    Saved to results/provider_models.json so the chosen model name can be
    checked against the provider's own list. Non-fatal: any failure is
    recorded and the run continues.
    """
    info = {"backend_url": backend_url, "requested_model": model}
    try:
        import urllib.request
        req = urllib.request.Request(
            backend_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        ids = [m.get("id") for m in data.get("data", [])
               if isinstance(m, dict)]
        info["models"] = ids
        info["model_listed"] = model in ids
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "provider_models.json").write_text(json.dumps(info, indent=2))
    print(f"Provider model probe: listed={info.get('model_listed')} "
          f"count={len(info.get('models', []))} error={info.get('error')}")


def run(ticker, date, provider, model):
    if not ticker:
        print("ERROR: --ticker is required in normal mode", file=sys.stderr)
        return 2
    provider = provider.lower()
    backend_url = None
    if provider in COMPAT_PROVIDERS:
        spec = COMPAT_PROVIDERS[provider]
        backend_url = spec["backend_url"]
        key = (os.environ.get(spec["key_env"])
               or os.environ.get("OPENAI_COMPATIBLE_API_KEY"))
        if not key:
            print(f"ERROR: {spec['key_env']} is not set. Add it as a "
                  f"repository secret named {spec['key_env']} before "
                  f"running in normal mode.", file=sys.stderr)
            return 2
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = key
        probe_models(backend_url, key, model)
        ta_provider = "openai_compatible"
    else:
        key_env = {
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider, "GOOGLE_API_KEY")
        if not os.environ.get(key_env):
            print(f"ERROR: {key_env} is not set. Add it as a repository secret "
                  f"named {key_env} before running in normal mode.",
                  file=sys.stderr)
            return 2
        ta_provider = provider
    if not date:
        date = date_cls.today().isoformat()

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = ta_provider
    if backend_url:
        config["backend_url"] = backend_url
    config["deep_think_llm"] = model
    config["quick_think_llm"] = model
    # Keep free-tier runs bounded: one debate / discussion round.
    for k in ("max_debate_rounds", "max_risk_discuss_rounds"):
        if k in config:
            config[k] = 1

    ta = TradingAgentsGraph(debug=True, config=config)
    out = ta.propagate(ticker, date)
    if isinstance(out, tuple) and len(out) == 2:
        final_state, decision = out
    else:
        final_state, decision = {}, out

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "decision.txt").write_text(str(decision))

    state_txt = {}
    if isinstance(final_state, dict):
        for k, v in final_state.items():
            state_txt[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    (RESULTS / "final_state.json").write_text(
        json.dumps(state_txt, indent=2, ensure_ascii=False))

    sections = [
        ("Market report", "market_report"),
        ("Sentiment report", "sentiment_report"),
        ("News report", "news_report"),
        ("Fundamentals report", "fundamentals_report"),
        ("Investment plan", "investment_plan"),
        ("Final trade decision", "final_trade_decision"),
    ]
    lines = [f"# TradingAgents report - {ticker} - {date}",
             f"Provider: {provider} - Model: {model}",
             f"Endpoint: {backend_url or 'provider default'}", "",
             "## Decision", "", str(decision), ""]
    for title, key in sections:
        if isinstance(final_state, dict) and final_state.get(key):
            lines += [f"## {title}", "", str(final_state[key]), ""]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print(f"Done. Decision: {decision}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        return smoke()
    return run(args.ticker, args.date, args.provider, args.model)


if __name__ == "__main__":
    sys.exit(main())
