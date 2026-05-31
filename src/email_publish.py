"""
Send today's MTG v2 snapshot as an HTML email via Resend.

Mirrors the delivery path of the sibling v1 project (Market Temp Gauge):
same Resend API, same env vars, same recipient. Loads credentials from
v1's .env file so there's a single source of truth for email config -- if
the user rotates the Resend key in v1, v2 picks it up automatically.

Env vars (read from MTG_V2_ENV_FILE or default to ../Market Temp Gauge/.env):
    RESEND_API_KEY      required; sender uses Resend's REST API
    REPORT_EMAIL_TO     required; primary recipient
    REPORT_EMAIL_FROM   required; must be a Resend-verified domain
    REPORT_EMAIL_BCC    optional; comma-separated

If RESEND_API_KEY or REPORT_EMAIL_TO are missing, returns
{"sent": False, "reason": ...} -- gracefully degrades like v1's sender.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENV_PATH = (
    Path(__file__).resolve().parent.parent.parent / "Market Temp Gauge" / ".env"
)


# ---------------------------------------------------------------------------
# env loader (no python-dotenv dep)
# ---------------------------------------------------------------------------

def load_env(path: Path | None = None) -> dict[str, str]:
    """
    Parse a .env file into a dict. Tolerates blank lines, comments, and
    optional `export ` prefix. Single/double quotes around values are stripped.
    """
    env_path = path or Path(os.environ.get("MTG_V2_ENV_FILE", str(DEFAULT_ENV_PATH)))
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        out[key] = val
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

# Lightweight per-regime palette. Not trying to match v1's 7-band scale
# byte-for-byte -- v2 has 6 bands and a different taxonomy, so we pick
# semantically aligned colors.
REGIME_COLORS = {
    "Panic":     {"bg": "#FFF0F0", "border": "#F5A0A0", "text": "#8B1A1A"},
    "Fear":      {"bg": "#FFF0E6", "border": "#E89866", "text": "#8B3A00"},
    "Anxiety":   {"bg": "#FFF5E6", "border": "#E8B866", "text": "#7A4A00"},
    "Neutral":   {"bg": "#F5F5F0", "border": "#CCC",    "text": "#555"},
    "Greed":     {"bg": "#FFF8E6", "border": "#E8C766", "text": "#7A5C00"},
    "Euphoria":  {"bg": "#F0FFF0", "border": "#90D090", "text": "#1A6B1A"},
}


def _esc(s) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_pct(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.2f}%"


def _fmt_prob(p) -> str:
    if p is None:
        return "n/a"
    return f"{p * 100:.0f}%"


REGIME_BLURB = {
    "Panic":     "an environment of acute stress -- volatility elevated, prices falling, with something specific breaking",
    "Fear":      "a stressed market environment with prices well below recent highs but no single breaking point",
    "Anxiety":   "a cautious environment -- elevated nervousness and a meaningful pullback from recent highs",
    "Neutral":   "an average market environment -- nothing notably calm or stressed",
    "Greed":     "a calm, confident environment -- prices near recent highs, low volatility",
    "Euphoria":  "an exceptionally calm and confident environment -- low volatility, prices at or near new highs",
}


def _human_date(iso: str) -> str:
    """'2000-01-21' -> '21 January 2000' (cross-platform; no %-d on Windows)."""
    from datetime import date
    try:
        d = date.fromisoformat(iso)
        return d.strftime("%d %B %Y").lstrip("0")
    except Exception:
        return iso


def _historical_context(analog_date: str) -> str | None:
    """
    A short, neutral one-line context note for famous analog dates.
    Only fires for genuinely well-known dates -- otherwise returns None and
    the email omits the context line. Conservative on commentary.
    """
    notes = {
        "2000": "near the peak of the dot-com bubble",
        "2007": "in the late stages of the pre-GFC bull market",
        "2008": "during the global financial crisis",
        "2011": "around the US debt downgrade and European sovereign debt crisis",
        "2018-01": "near the early-2018 market peak",
        "2018-12": "near the late-2018 correction low",
        "2020-02": "as the COVID sell-off was beginning",
        "2020-03": "at the depths of the COVID crash",
        "2022": "during the 2022 bear market",
    }
    if analog_date.startswith("2018-01"):
        return notes["2018-01"]
    if analog_date.startswith("2018-12"):
        return notes["2018-12"]
    if analog_date.startswith("2020-02"):
        return notes["2020-02"]
    if analog_date.startswith("2020-03"):
        return notes["2020-03"]
    year = analog_date[:4]
    return notes.get(year)


def _round_pct_words(p: float | None) -> str:
    """0.36 -> 'about one in three (36%)'."""
    if p is None:
        return "not enough information to estimate"
    pct = round(p * 100)
    if pct <= 5:
        return f"very rare ({pct}%)"
    if pct <= 15:
        return f"around one in ten ({pct}%)"
    if pct <= 25:
        return f"around one in five ({pct}%)"
    if pct <= 38:
        return f"about one in three ({pct}%)"
    if pct <= 55:
        return f"around half ({pct}%)"
    if pct <= 65:
        return f"more than half ({pct}%)"
    if pct <= 80:
        return f"around three in four ({pct}%)"
    return f"the large majority ({pct}%)"


def _reflection_html(reflect: dict | None) -> str:
    """
    A small highlighted strip recapping what the gauge said 3 months ago and
    how it landed, in a single tight line. Empty if reflection isn't available
    (cold start, or realised 90d return isn't in yet).

    The actual return is the close-to-close S&P 500 return from reflect_ts to
    as_of_ts, computed from yfinance dailies. No rounding beyond display.
    """
    if not reflect:
        return ""

    reflect_date = _human_date(reflect["date"])
    regime = reflect["regime"]
    rc = REGIME_COLORS.get(regime, REGIME_COLORS["Neutral"])
    pred = reflect.get("predicted_90d", {})
    p25, p75 = pred.get("p25"), pred.get("p75")
    actual = reflect.get("actual_90d")

    def _fmt(v: float | None) -> str:
        return f"{v:+.1f}%" if v is not None else "n/a"

    parts = [
        f'<strong>{_esc(reflect_date)}</strong>: '
        f'<strong style="color:{rc["text"]};">{_esc(regime)}</strong>.'
    ]
    if p25 is not None and p75 is not None:
        parts.append(f"History said <strong>{_fmt(p25)}</strong> to <strong>{_fmt(p75)}</strong>;")
    if actual is not None:
        parts.append(f"actually <strong>{_fmt(actual)}</strong>.")

    body = " ".join(parts)

    return f"""
<tr><td style="padding:0 28px 14px 28px;">
<div style="background:#faf8f2;border:1px solid #ece6d4;border-left:4px solid {rc['border']};border-radius:4px;padding:10px 16px;font-size:13px;color:#444;line-height:1.55;">
<div style="font-size:10px;color:#999;text-transform:uppercase;letter-spacing:1.3px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin-bottom:3px;">Looking back &middot; 3 months ago</div>
{body}
</div>
</td></tr>"""


def render_html(summary: dict) -> str:
    """
    Render today.json into a CLIENT-FRIENDLY HTML email.

    Design rules:
      - No jargon: no Brier, ECE, OOS, cosine, k-means, percentile, base rate.
      - Lead with the regime word + one plain-English sentence.
      - Show the single most powerful piece of information (the closest
        historical match) with neutral context.
      - Translate distributions into plain phrases ("around one in three"),
        not percentile tables.
      - Honest framing throughout: "history says... not a forecast."
      - One short bottom-line paragraph the reader can act on conceptually.
    """
    q = summary["query_features"]
    r = summary["regime"]
    c = summary["closest_analog"]
    br = summary["base_rates"]
    reflection_html = _reflection_html(summary.get("reflection"))

    rc = REGIME_COLORS.get(r["label"], REGIME_COLORS["Neutral"])
    as_of_human = _human_date(summary["as_of_date"])
    regime_label = r["label"]
    regime_blurb = REGIME_BLURB.get(regime_label, "")

    # Closest analog narrative
    analog_human = _human_date(c["date"])
    analog_context = _historical_context(c["date"])
    co = c["outcomes"]
    fwd12 = co.get("fwd_ret_12m")
    fwd12_clause = ""
    if fwd12 is not None:
        if fwd12 < -3:
            fwd12_clause = f", and went on to lose {abs(fwd12):.0f}% over the year that followed"
        elif fwd12 > 3:
            fwd12_clause = f", and went on to gain {fwd12:.0f}% over the year that followed"
        else:
            fwd12_clause = f", and was roughly flat over the year that followed ({fwd12:+.1f}%)"

    analog_para = (
        f"<strong>Today most closely resembles {analog_human}</strong>"
        + (f" &mdash; {analog_context}" if analog_context else "")
        + f". The match across the four factors we track is unusually tight"
        + fwd12_clause + "."
    )

    # Plain-English summary of 30d and 90d outcomes from the 20-analog set.
    ret30 = br["returns"]["fwd_ret_30d"]
    ret90 = br["returns"]["fwd_ret_90d"]
    dd90 = br["drawdowns"]["max_dd_90d"]
    p_loss_30 = (ret30.get("p_below", {}).get("+0%") or {}).get("p")
    p_loss_90 = (ret90.get("p_below", {}).get("+0%") or {}).get("p")
    p_dd20_90 = (dd90.get("p_below", {}).get("-20%") or {}).get("p")

    def trend_word(median: float | None) -> str:
        if median is None:
            return "mixed"
        if median > 1.5:
            return "modestly higher"
        if median > 0.3:
            return "slightly higher"
        if median > -0.3:
            return "roughly flat"
        if median > -1.5:
            return "slightly lower"
        return "modestly lower"

    p25_30 = ret30["stats"].get("p25")
    p75_30 = ret30["stats"].get("p75")
    p25_90 = ret90["stats"].get("p25")
    p75_90 = ret90["stats"].get("p75")

    def range_phrase(p25, p75):
        if p25 is None or p75 is None:
            return ""
        lo = abs(p25) if p25 < 0 else p25
        hi = abs(p75) if p75 < 0 else p75
        lo_sign = "a {:.0f}% loss".format(lo) if p25 < 0 else "a {:.0f}% gain".format(lo)
        hi_sign = "a {:.0f}% loss".format(hi) if p75 < 0 else "a {:.0f}% gain".format(hi)
        return f"with typical outcomes ranging from {lo_sign} to {hi_sign}"

    history_html = (
        '<p style="margin:0 0 10px 0;font-size:14px;color:#333;line-height:1.55;">'
        f"Across the 20 historical periods most similar to today (since 1990): "
        f"over the next month, markets were typically <strong>{trend_word(ret30['stats'].get('median'))}</strong>, "
        f"{range_phrase(p25_30, p75_30)}. Around {_round_pct_words(p_loss_30)} of similar periods ended the month lower."
        "</p>"
        '<p style="margin:0 0 10px 0;font-size:14px;color:#333;line-height:1.55;">'
        f"Over the next three months, the pattern was similar: typically <strong>{trend_word(ret90['stats'].get('median'))}</strong>, "
        f"{range_phrase(p25_90, p75_90)}. Severe declines (more than 20% in three months) "
        f"from these conditions were {_round_pct_words(p_dd20_90)}."
        "</p>"
        '<p style="margin:0 0 0 0;font-size:14px;color:#333;line-height:1.55;">'
        "Over the next twelve months, history was too varied to summarise simply &mdash; "
        "outcomes from similar starting points ranged widely, and we don't draw a confident "
        "line at that horizon."
        "</p>"
    )

    # State line in plain English (no VIX/DD jargon-style labels)
    vix_descr = (
        "low" if q["vix"] < 15 else
        "moderate" if q["vix"] < 22 else
        "elevated" if q["vix"] < 30 else
        "high"
    )
    state_phrase = (
        f"Volatility is <strong>{vix_descr}</strong> (VIX {q['vix']:.1f}). "
        f"The S&amp;P 500 is "
        f"{('up' if q['spx_ytd_pct'] >= 0 else 'down')} "
        f"{abs(q['spx_ytd_pct']):.1f}% for the year "
        f"and {abs(q['spx_dd_from_52w']):.1f}% below its 12-month high."
    )

    return f"""<!doctype html>
<html><body style="margin:0;padding:0;background:#fafafa;font-family:Georgia,'Times New Roman',serif;color:#222;">
<table cellpadding="0" cellspacing="0" border="0" width="100%" style="background:#fafafa;">
<tr><td align="center" style="padding:28px 12px;">
<table cellpadding="0" cellspacing="0" border="0" width="640" style="background:#fff;border:1px solid #e5e5e5;border-radius:6px;max-width:640px;">

<tr><td style="padding:24px 28px 4px 28px;">
<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1.5px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">Daily market temperature</div>
<div style="font-size:14px;color:#666;margin-top:4px;">{_esc(as_of_human)}</div>
</td></tr>

<tr><td style="padding:14px 28px 18px 28px;">
<div style="background:{rc['bg']};border:1px solid {rc['border']};border-radius:6px;padding:20px 24px;text-align:center;">
<div style="font-size:11px;color:{rc['text']};text-transform:uppercase;letter-spacing:1.8px;opacity:0.7;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">Today's reading</div>
<div style="font-size:36px;font-weight:700;color:{rc['text']};margin:8px 0 6px 0;font-family:Georgia,serif;">{_esc(regime_label)}</div>
<div style="font-size:14px;color:{rc['text']};opacity:0.85;font-style:italic;">{_esc(regime_blurb)}</div>
</div>
</td></tr>
{reflection_html}
<tr><td style="padding:0 28px 16px 28px;font-size:14px;color:#333;line-height:1.55;">
{state_phrase}
</td></tr>

<tr><td style="padding:8px 28px 16px 28px;font-size:14px;color:#222;line-height:1.6;border-top:1px solid #eee;padding-top:18px;">
{analog_para}
</td></tr>

<tr><td style="padding:0 28px 18px 28px;border-top:1px solid #eee;padding-top:16px;">
<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">What history says next</div>
{history_html}
</td></tr>

<tr><td style="padding:14px 28px 22px 28px;border-top:1px solid #eee;font-size:12px;color:#777;line-height:1.55;font-style:italic;">
This report describes what has happened in history from market conditions similar to today's.
It is not a forecast and should not be read as one. Outcomes from any single historical
period varied widely; today's reading is one input to ongoing judgement, not a prediction
of returns.
</td></tr>

</table>
</td></tr>
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Send via Resend
# ---------------------------------------------------------------------------

def send_via_resend(
    *,
    api_key: str,
    sender: str,
    recipient: str,
    subject: str,
    html: str,
    bcc: list[str] | None = None,
) -> dict:
    body = {"from": sender, "to": [recipient], "subject": subject, "html": html}
    if bcc:
        body["bcc"] = bcc
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare in front of Resend returns 403 (error 1010) for the
            # default urllib User-Agent. Mimic a normal client.
            "User-Agent": "mtg-v2/0.1 (+Python urllib)",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return {"sent": True, "id": payload.get("id"), "to": recipient}
    except urllib.error.HTTPError as e:
        return {
            "sent": False,
            "reason": f"Resend {e.code}: {e.read().decode('utf-8', errors='replace')}",
            "to": recipient,
        }
    except Exception as e:
        return {"sent": False, "reason": f"{type(e).__name__}: {e}", "to": recipient}


def send_snapshot_email(summary: dict, env: dict[str, str] | None = None) -> dict:
    """
    Render and send today's snapshot email. Gracefully degrades:
    if Resend isn't configured, returns {"sent": False, "reason": ...}.
    """
    env = env if env is not None else load_env()
    api_key = env.get("RESEND_API_KEY") or os.environ.get("RESEND_API_KEY")
    sender = env.get("REPORT_EMAIL_FROM") or os.environ.get("REPORT_EMAIL_FROM")
    recipient = env.get("REPORT_EMAIL_TO") or os.environ.get("REPORT_EMAIL_TO")
    bcc_raw = env.get("REPORT_EMAIL_BCC") or os.environ.get("REPORT_EMAIL_BCC", "")
    bcc = [s.strip() for s in bcc_raw.split(",") if s.strip()]

    if not api_key:
        return {"sent": False, "reason": "RESEND_API_KEY not set (checked v1 .env and os.environ)"}
    if not sender:
        return {"sent": False, "reason": "REPORT_EMAIL_FROM not set"}
    if not recipient:
        return {"sent": False, "reason": "REPORT_EMAIL_TO not set"}

    html = render_html(summary)
    regime = summary["regime"]["label"]
    analog_human = _human_date(summary["closest_analog"]["date"])
    as_of_human = _human_date(summary["as_of_date"])
    subject = f"Market temperature — {as_of_human} — {regime} (resembles {analog_human})"
    return send_via_resend(
        api_key=api_key, sender=sender, recipient=recipient,
        subject=subject, html=html, bcc=bcc,
    )
