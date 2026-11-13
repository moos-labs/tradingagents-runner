#!/usr/bin/env python3
"""Fleet context for the Borsagator agent runners.

The fleet's own verified data layer, shared by the agent tools so they
reason over what the fleet actually collected instead of vendor feeds
that silently return nothing for whole markets (yfinance carries no
EGX news and no usable POUL fundamentals — the agents' news and
fundamentals reports came back empty while their verdicts still voted).

Sources (all free, no keys):
- Google News RSS (Arabic + English). For EGX tickers this also
  surfaces Mubasher's republication of the official EGX disclosures.
- SEC EDGAR (US tickers): submissions list + company-concept facts.
- A curated statements CSV in the repo's data/ directory, named
  data/<code>_statements.csv (e.g. data/poul_statements.csv), when one
  exists — the same verified file the fundamentals runner computes
  ratios from.

Point-in-time discipline: items dated after the analysis date are
dropped. Everything fetched is cached per process and can be saved to
results/ so the sweep archive preserves exactly what the agent saw.
"""
import json, os, re, time, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

UA = {"User-Agent": "Moos Labs research (moos@sherbeeny.com)"}

ALIASES = {
    "POUL.CA": {"ar": "القاهرة للدواجن", "en": "Cairo Poultry"},
    "COMI.CA": {"ar": "البنك التجاري الدولي", "en": "Commercial International Bank Egypt"},
    "ETEL.CA": {"ar": "المصرية للاتصالات", "en": "Telecom Egypt"},
    "HRHO.CA": {"ar": "إي إف جي هيرميس", "en": "EFG Hermes"},
    "TMGH.CA": {"ar": "طلعت مصطفى", "en": "Talaat Moustafa Group"},
    "SWDY.CA": {"ar": "السويدي إلكتريك", "en": "Elsewedy Electric"},
    "ESRS.CA": {"ar": "حديد عز", "en": "Ezz Steel"},
    "ABUK.CA": {"ar": "أبو قير للأسمدة", "en": "Abu Qir Fertilizers"},
    "AMOC.CA": {"ar": "الإسكندرية للزيوت المعدنية", "en": "AMOC Egypt"},
    "EKHO.CA": {"ar": "القابضة المصرية الكويتية", "en": "Egypt Kuwait Holding"},
}

MAJOR_FORMS = {"10-K", "10-Q", "8-K", "20-F", "6-K", "DEF 14A", "S-1",
               "424B2", "FWP", "SC 13G", "SC 13D", "4"}

FACT_CONCEPTS = [
    ("Revenues", "Revenue"),
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
    ("NetIncomeLoss", "Net income"),
    ("EarningsPerShareDiluted", "EPS (diluted)"),
    ("Assets", "Total assets"),
    ("StockholdersEquity", "Total equity"),
    ("NetCashProvidedByUsedInOperatingActivities", "Operating cash flow"),
]


def _pdate(s):
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            return datetime.strptime(s or "", fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


def _is_arabic(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in text or "")


def _gnews(query, lang, country, days):
    q = f"{query} when:{days}d"
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
           + f"&hl={lang}&gl={country}&ceid={country}:{lang}")
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        src = it.find("source")
        items.append({
            "title": (it.findtext("title") or "").strip(),
            "date": it.findtext("pubDate") or "",
            "source": (src.text if src is not None else "").strip(),
            "url": it.findtext("link") or "",
        })
    return items


def _classify_egx(item, code):
    t = item["title"]
    if code and code in t:
        return "disclosure"
    if t.startswith("العنوان"):
        return "disclosure"
    return "news"


def fetch_rss(ticker, days=30):
    code = ticker.upper()
    egx = code.endswith(".CA")
    alias = ALIASES.get(code, {})
    queries = []
    if egx:
        if alias.get("ar"):
            queries.append((alias["ar"], "ar", "EG"))
        queries.append((code, "ar", "EG"))
        if alias.get("en"):
            queries.append((alias["en"], "en", "US"))
        queries.append((code + " stock", "en", "US"))
    else:
        bare = code.split(".")[0]
        queries.append((f"{bare} stock", "en", "US"))
        queries.append((f'"{bare}" earnings OR filing OR SEC', "en", "US"))
    seen, items = set(), []
    for q, lang, country in queries:
        try:
            got = _gnews(q, lang, country, days)
        except Exception:
            continue
        for it in got:
            key = re.sub(r"\s+", " ",
                         re.sub(r" - [^-]+$", "", it["title"])).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            it["kind"] = _classify_egx(it, code) if egx else "news"
            items.append(it)
    if egx:
        # Arabic relevance guard: bare-code queries transliterate in
        # Arabic Google News (POUL -> بول "Paul"), flooding the feed
        # with unrelated people news. Arabic items must mention the
        # company's distinctive Arabic name token or the Latin code.
        alias_ar = (ALIASES.get(code) or {}).get("ar") or ""
        tokens = [w for w in re.split(r"\s+", alias_ar) if len(w) >= 5]
        if tokens:
            items = [i for i in items
                     if not _is_arabic(i["title"])
                     or any(t in i["title"] for t in tokens)
                     or code in i["title"]]
    items.sort(key=lambda i: _pdate(i["date"]), reverse=True)
    return items


def _cik_for(ticker):
    bare = ticker.upper().split(".")[0]
    m = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=UA, timeout=30).json()
    for row in m.values():
        if row["ticker"] == bare:
            return str(row["cik_str"]).zfill(10), row.get("title", bare)
    return None, None


def fetch_edgar_filings(ticker, limit=25):
    try:
        cik, _name = _cik_for(ticker)
        if not cik:
            return []
        d = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         headers=UA, timeout=30).json()
        rec = d["filings"]["recent"]
        out = []
        for i in range(len(rec["form"])):
            if rec["form"][i] not in MAJOR_FORMS:
                continue
            acc = rec["accessionNumber"][i].replace("-", "")
            out.append({
                "title": f"{rec['form'][i]} — {d.get('name', ticker)}",
                "date": rec["filingDate"][i],
                "source": "SEC EDGAR",
                "url": (f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(cik)}/{acc}/{rec['primaryDocument'][i]}"),
                "kind": "disclosure",
            })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def fetch_edgar_facts(ticker, as_of):
    """Latest annual + quarterly values for the headline concepts,
    counting only filings made on or before the analysis date."""
    try:
        cik, name = _cik_for(ticker)
        if not cik:
            return None
        lines = [f"SEC EDGAR company facts — {name} ({ticker.upper()})"]
        got_any = False
        for concept, label in FACT_CONCEPTS:
            try:
                d = requests.get(
                    f"https://data.sec.gov/api/xbrl/companyconcept/"
                    f"CIK{cik}/us-gaap/{concept}.json",
                    headers=UA, timeout=30).json()
            except Exception:
                continue
            units = d.get("units") or {}
            entries = []
            for unit_rows in units.values():
                entries.extend(unit_rows)
            entries = [e for e in entries
                       if (e.get("filed") or "9999") <= as_of
                       and e.get("form") in ("10-K", "20-F", "10-Q", "6-K")]
            if not entries:
                continue
            entries.sort(key=lambda e: e.get("filed", ""), reverse=True)
            picks, seen_forms = [], set()
            for e in entries:
                fam = "annual" if e["form"] in ("10-K", "20-F") else "quarterly"
                if fam in seen_forms:
                    continue
                seen_forms.add(fam)
                picks.append(e)
                if len(picks) >= 2:
                    break
            for e in picks:
                val = e.get("val")
                if isinstance(val, (int, float)) and abs(val) >= 1e6:
                    val = f"{val:,.0f}"
                lines.append(
                    f"- {label} ({e.get('form')}, period ended {e.get('end')}, "
                    f"filed {e.get('filed')}): {val} {e.get('unit', '')}".rstrip())
                got_any = True
        return "\n".join(lines) if got_any else None
    except Exception:
        return None


def fetch_ohlc(ticker, as_of):
    """Daily OHLC history from Yahoo's chart API directly — the fleet's
    standard price route (period-based requests: range=max silently
    degrades to monthly bars). yfinance cannot read .CA tickers at all
    (Egypt DST bug) and its long POUL history mixes the pre-merger
    price scale into the series.

    The returned series is CONTINUOUS — corporate-action resets are
    back-adjusted the way data vendors adjust for splits:
    - Yahoo split events (when the API reports them) are applied.
    - On price-limit markets (.CA: EGX moves are capped at ±20%), any
      residual overnight close jump beyond ±25% cannot be a trade — it
      is an exchange reference-price reset (e.g. POUL's Sep-2026
      merger, 38.20 -> 9.71). Earlier rows are scaled by the break
      ratio so returns and moving averages stay meaningful; volumes
      are scaled by the inverse.
    Returns a list of row dicts or None."""
    try:
        end = datetime.strptime(as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        end = datetime.now(timezone.utc)
    p2 = int(end.timestamp()) + 2 * 86400
    p1 = int(end.timestamp()) - 5 * 365 * 86400
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(ticker))
    data = None
    for attempt in (1, 2):
        try:
            r = requests.get(url, params={"period1": p1, "period2": p2,
                                          "interval": "1d",
                                          "events": "split"},
                             headers=UA, timeout=30)
            r.raise_for_status()
            res = (r.json().get("chart") or {}).get("result") or []
            if res:
                data = res[0]
                break
        except Exception:
            if attempt == 2:
                return None
            time.sleep(3)
    if data is None:
        return None
    off = int(((data.get("meta") or {}).get("gmtoffset") or 0))
    quote = ((data.get("indicators") or {}).get("quote") or [{}])[0]

    def g(key, i):
        arr = quote.get(key) or []
        return arr[i] if i < len(arr) else None

    rows = []
    for i, t in enumerate(data.get("timestamp") or []):
        close = g("close", i)
        if close is None:
            continue
        rows.append({
            "date": datetime.fromtimestamp(t + off, timezone.utc)
                    .strftime("%Y-%m-%d"),
            "open": g("open", i), "high": g("high", i),
            "low": g("low", i), "close": close,
            "volume": g("volume", i),
            "_f": 1.0,
        })
    if not rows:
        return None
    # Yahoo split events: "2:1" on date D -> pre-D prices x 1/2.
    splits = ((data.get("events") or {}).get("splits") or {})
    for ev in splits.values():
        try:
            num, den = (float(x) for x in str(ev.get("splitRatio", ""))
                        .split(":"))
            d0 = datetime.fromtimestamp(int(ev.get("date", 0)) + off,
                                        timezone.utc).strftime("%Y-%m-%d")
            for row in rows:
                if row["date"] < d0:
                    row["_f"] *= den / num if num else 1.0
        except Exception:
            continue
    # Reference-price resets on price-limit markets (see docstring):
    # scan backwards, carrying the cumulative adjustment factor.
    if ticker.upper().endswith(".CA"):
        f = 1.0
        for i in range(len(rows) - 1, 0, -1):
            rows[i]["_f"] *= f
            prev, cur = rows[i - 1]["close"], rows[i]["close"]
            if prev:
                ratio = cur / prev
                if ratio > 1.25 or ratio < 0.75:
                    f *= ratio
        rows[0]["_f"] *= f
    for row in rows:
        f = row.pop("_f")
        for k in ("open", "high", "low", "close"):
            if row[k] is not None:
                row[k] = row[k] * f
        row["adj"] = row["close"]
        if row["volume"]:
            row["volume"] = row["volume"] / f if f else row["volume"]
    return rows


class FleetContext:
    """Fetch-once cache of everything the fleet knows about a ticker."""

    def __init__(self, ticker, as_of, days=30):
        self.ticker = ticker.upper()
        self.as_of = as_of  # 'YYYY-MM-DD'
        self.days = days
        self._items = None
        self._fundamentals = None
        self._fundamentals_done = False
        self._ohlc_rows = None

    def ohlc_csv(self, start_date, end_date):
        """OHLCV in the exact CSV shape yfinance's vendor returns, so
        TradingAgents' market analyst reads the fleet's price series."""
        if self._ohlc_rows is None:
            self._ohlc_rows = fetch_ohlc(self.ticker, self.as_of) or []
        if not self._ohlc_rows:
            return None

        def n(v):
            if v is None:
                return ""
            return str(int(v)) if float(v) == int(v) and abs(v) >= 1000 \
                else f"{v:.4f}".rstrip("0").rstrip(".")

        lines = ["Date,Open,High,Low,Close,Adj Close,Volume"]
        for r in self._ohlc_rows:
            if start_date <= r["date"] <= end_date:
                lines.append(",".join([
                    r["date"], n(r["open"]), n(r["high"]), n(r["low"]),
                    n(r["close"]), n(r["adj"]),
                    str(int(r["volume"])) if r["volume"] else "0"]))
        return "\n".join(lines) if len(lines) > 1 else None

    def indicator_text(self, indicator, as_of_date, look_back_days):
        """One technical indicator over a lookback window, computed by
        stockstats (the same engine TradingAgents' vendor uses) on the
        fleet's adjusted series — so the market analyst's SMA/RSI/MACD
        values stop coming from the broken vendor series."""
        if self._ohlc_rows is None:
            self._ohlc_rows = fetch_ohlc(self.ticker, self.as_of) or []
        if not self._ohlc_rows:
            return None
        try:
            import pandas as pd
            from stockstats import StockDataFrame
            rows = [r for r in self._ohlc_rows if r["date"] <= as_of_date]
            if len(rows) < 30:
                return None
            df = pd.DataFrame({
                "date": [r["date"] for r in rows],
                "open": [r["open"] for r in rows],
                "high": [r["high"] for r in rows],
                "low": [r["low"] for r in rows],
                "close": [r["close"] for r in rows],
                "volume": [r["volume"] or 0 for r in rows],
            })
            sdf = StockDataFrame.retype(df.set_index("date"))
            series = sdf[indicator]
            tail = [(d, v) for d, v in zip(df["date"], series)
                    if v == v]  # drop NaN warm-up rows
            if not tail:
                return None
            tail = tail[-max(1, int(look_back_days)):]
            lines = [f"{indicator} for {self.ticker} up to {as_of_date} "
                     f"(fleet-adjusted price series):"]
            lines += [f"{d}: {v:.4f}" for d, v in tail]
            return "\n".join(lines)
        except Exception:
            return None

    def items(self):
        if self._items is None:
            items = fetch_rss(self.ticker, self.days)
            if not self.ticker.endswith(".CA"):
                items = fetch_edgar_filings(self.ticker) + items
            as_of_dt = datetime.strptime(self.as_of, "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
            self._items = [i for i in items if _pdate(i["date"]) <= as_of_dt]
        return self._items

    def news_text(self, max_items=40, max_chars=9000):
        items = self.items()
        if not items:
            return None
        disc = [i for i in items if i["kind"] == "disclosure"]
        news = [i for i in items if i["kind"] != "disclosure"]
        lines = [
            f"Fleet-verified news & disclosures for {self.ticker} "
            f"(fetched live by the Borsagator fleet from Google News RSS"
            + ("" if self.ticker.endswith(".CA") else " and SEC EDGAR")
            + f"; nothing after {self.as_of}):",
        ]
        if disc:
            lines.append("\nOFFICIAL DISCLOSURES / FILINGS:")
            for i in disc[:25]:
                lines.append(f"- [{i['date']}] ({i['source']}) {i['title']}")
        if news:
            lines.append("\nNEWS COVERAGE:")
            for i in news[:max_items]:
                lines.append(f"- [{i['date']}] ({i['source']}) {i['title']}")
        text = "\n".join(lines)
        return text[:max_chars]

    def fundamentals_text(self):
        if not self._fundamentals_done:
            self._fundamentals_done = True
            if not self.ticker.endswith(".CA"):
                self._fundamentals = fetch_edgar_facts(self.ticker, self.as_of)
            if not self._fundamentals:
                code = self.ticker.split(".")[0].lower()
                for cand in (f"data/{code}_statements.csv",
                             f"data/{self.ticker.lower()}_statements.csv"):
                    if os.path.exists(cand):
                        with open(cand, encoding="utf-8") as f:
                            body = f.read()
                        self._fundamentals = (
                            f"Fleet-verified financial statements for "
                            f"{self.ticker} (curated CSV, EGP millions unless "
                            f"stated; CURRENT_NOV2026 column is a memo "
                            f"snapshot):\n{body}")
                        break
        return self._fundamentals

    def save(self, results_dir):
        os.makedirs(results_dir, exist_ok=True)
        payload = {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "items": self.items(),
            "fundamentals": self.fundamentals_text(),
        }
        with open(os.path.join(results_dir, "fleet_context.json"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        text = self.news_text() or "(no news items fetched)"
        with open(os.path.join(results_dir, "fleet_context.md"), "w",
                  encoding="utf-8") as f:
            f.write(f"# Fleet context — {self.ticker} — as of {self.as_of}\n\n"
                    f"{text}\n\n## Fundamentals\n\n"
                    f"{self.fundamentals_text() or '(none available)'}\n")
