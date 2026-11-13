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
  cohere        COHERE_API_KEY      (same generic client pointed at
                                     Cohere's OpenAI-compatibility endpoint
                                     api.cohere.ai/compatibility/v1)
  huggingface   HF_TOKEN            (same generic client pointed at the
                                     Hugging Face router
                                     router.huggingface.co/v1)
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
    "cohere": {
        "backend_url": "https://api.cohere.ai/compatibility/v1",
        "key_env": "COHERE_API_KEY",
    },
    "huggingface": {
        "backend_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
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
                        "sambanova, github-models, cohere or huggingface "
                        "(default: google)")
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


# ---------------------------------------------------------------------------
# Cohere tool-schema patch. Cohere's OpenAI-compatibility endpoint validates
# tool schemas strictly and rejects any property without a JSON-schema
# "type" (observed failure: 400 "property look_back_days must have a
# type"). TradingAgents defines some tool parameters without type
# annotations, so LangChain emits properties with no "type". Before the
# graph runs on the cohere provider, two generic passes add the missing
# types: (A) rebuild every TradingAgents tool's args_schema with inferred
# field types, and (B) wrap LangChain's OpenAI tool conversion so every
# emitted property schema carries a "type".
# ---------------------------------------------------------------------------

_INTEGER_PROPS = {"look_back_days", "lookback_days", "limit", "days",
                  "window", "period", "top_n", "count"}


def _infer_json_type(prop, name=None):
    if name in _INTEGER_PROPS:
        return "integer"
    if isinstance(prop, dict):
        any_of = prop.get("anyOf")
        if isinstance(any_of, list):
            for branch in any_of:
                if isinstance(branch, dict):
                    t = branch.get("type")
                    if t and t != "null":
                        return t
        default = prop.get("default")
        if isinstance(default, bool):
            return "boolean"
        if isinstance(default, int):
            return "integer"
        if isinstance(default, float):
            return "number"
        if isinstance(default, list):
            return "array"
        if isinstance(default, dict):
            return "object"
    return "string"


def _fix_json_schema(node):
    """Recursively add a JSON-schema "type" wherever one is missing."""
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            for name, prop in props.items():
                if isinstance(prop, dict):
                    if "type" not in prop:
                        prop["type"] = _infer_json_type(prop, name)
                        prop.pop("anyOf", None)
                    _fix_json_schema(prop)
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                _fix_json_schema(node[key])
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(node.get(key), list):
                for sub in node[key]:
                    _fix_json_schema(sub)
    return node


def _wrap_openai_converter(fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        try:
            if isinstance(out, dict):
                target = out
                if isinstance(out.get("function"), dict):
                    target = out["function"]
                params = target.get("parameters")
                if isinstance(params, dict):
                    _fix_json_schema(params)
                    params.setdefault("type", "object")
        except Exception:  # noqa: BLE001 - the patch must never break a run
            pass
        return out
    return wrapped


def _patch_tool_args_schemas():
    """Rebuild each TradingAgents tool's args_schema with typed fields."""
    import typing
    import pydantic
    from langchain_core.tools import BaseTool

    tools = {}
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("tradingagents"):
            continue
        try:
            candidates = list(vars(mod).values())
            for val in list(vars(mod).values()):
                if isinstance(val, type):
                    try:
                        candidates.extend(vars(val).values())
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            continue
        for val in candidates:
            if isinstance(val, BaseTool):
                tools[id(val)] = val
    patched = 0
    for tool in tools.values():
        schema = getattr(tool, "args_schema", None)
        if schema is None or not hasattr(schema, "model_fields"):
            continue
        fields = {}
        changed = False
        for fname, field in schema.model_fields.items():
            ann = field.annotation
            if ann is None or ann is typing.Any:
                default = field.default
                if fname in _INTEGER_PROPS:
                    ann = int
                elif isinstance(default, bool):
                    ann = bool
                elif isinstance(default, int):
                    ann = int
                elif isinstance(default, float):
                    ann = float
                elif isinstance(default, list):
                    ann = list
                elif isinstance(default, dict):
                    ann = dict
                else:
                    ann = str
                changed = True
            if field.is_required():
                fields[fname] = (ann, ...)
            else:
                fields[fname] = (ann, field.default)
        if changed:
            try:
                tool.args_schema = pydantic.create_model(
                    schema.__name__, **fields)
                patched += 1
            except Exception:  # noqa: BLE001
                pass
    print(f"Cohere schema patch: rebuilt args_schema for {patched} tools "
          f"({len(tools)} TradingAgents tools found)")


def patch_tool_schemas_for_cohere():
    _patch_tool_args_schemas()
    wrapped = []
    try:
        import langchain_core.utils.function_calling as fc
        for name in ("convert_to_openai_tool", "convert_to_openai_function"):
            if hasattr(fc, name):
                setattr(fc, name, _wrap_openai_converter(getattr(fc, name)))
                wrapped.append(f"langchain_core.{name}")
    except Exception:  # noqa: BLE001
        pass
    try:
        import langchain_openai.chat_models.base as ob
        for name in ("convert_to_openai_tool", "convert_to_openai_function"):
            if hasattr(ob, name):
                setattr(ob, name, _wrap_openai_converter(getattr(ob, name)))
                wrapped.append(f"langchain_openai.{name}")
    except Exception:  # noqa: BLE001
        pass
    print(f"Cohere schema patch: wrapped converters {wrapped}")


def install_fleet_vendor(ticker, analysis_date):
    """Register the fleet's own data layer as TradingAgents' FIRST news
    and fundamentals vendor (the framework's own vendor architecture).

    Root fix: the stock vendors carry no EGX news and no usable POUL
    fundamentals, and they return empty strings *successfully* — so the
    fallback chain never engaged and the news/fundamentals analyst
    reports came back empty while the verdict still voted. The fleet
    vendor serves what the fleet itself verified (live headlines +
    official disclosures + EDGAR facts / the curated statements CSV);
    when it has nothing it raises VendorUnavailableError, so the chain
    falls through to the stock vendors exactly as designed upstream.
    Returns the FleetContext (already fetched), or None if unavailable.
    """
    try:
        import fleet_context
        ctx = fleet_context.FleetContext(ticker, analysis_date)
        # Fetch eagerly: proves availability before the graph starts and
        # leaves a copy in results/ for the sweep archive.
        has_news = bool(ctx.items())
        has_fund = bool(ctx.fundamentals_text())
        RESULTS.mkdir(parents=True, exist_ok=True)
        ctx.save(str(RESULTS))
        print(f"Fleet vendor: news_items={len(ctx.items())} "
              f"fundamentals={'yes' if has_fund else 'no'}")
        if not has_news and not has_fund:
            return None

        from tradingagents.dataflows import router
        from tradingagents.dataflows.errors import VendorUnavailableError

        def fleet_get_news(t, start_date=None, end_date=None):
            text = ctx.news_text()
            if not text:
                raise VendorUnavailableError("fleet: no news items")
            return text

        def fleet_get_fundamentals(t, as_of_date=None):
            text = ctx.fundamentals_text()
            if not text:
                raise VendorUnavailableError("fleet: no fundamentals")
            return text

        if has_news:
            router.VENDOR_METHODS["get_news"]["fleet"] = fleet_get_news
        if has_fund:
            router.VENDOR_METHODS["get_fundamentals"]["fleet"] = \
                fleet_get_fundamentals
        return ctx
    except Exception as e:  # noqa: BLE001 - never break a run over context
        print(f"Fleet vendor unavailable ({type(e).__name__}: {e}); "
              f"continuing with stock vendors only")
        return None


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

    if provider == "cohere":
        patch_tool_schemas_for_cohere()

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

    # Fleet data first for news + fundamentals (see install_fleet_vendor).
    if install_fleet_vendor(ticker, date) is not None:
        vendors = dict(config.get("data_vendors") or {})
        vendors["news_data"] = "fleet,yfinance"
        vendors["fundamental_data"] = "fleet,sec_edgar,yfinance"
        config["data_vendors"] = vendors

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
