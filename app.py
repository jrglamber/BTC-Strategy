import os
import io
import json
import math
import zipfile
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response, redirect, url_for

try:
    import yfinance as yf
except Exception:
    yf = None

APP_NAME = "BTC Regime Research Logger"
APP_VERSION = "v5_webhook_secret_payload"
DB_PATH = os.getenv("DB_PATH", "/data/btc_research.sqlite" if os.path.exists("/data") else "btc_research.sqlite")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
BOOTSTRAP_ON_START = os.getenv("BOOTSTRAP_ON_START", "true").lower() in ("1", "true", "yes", "y")
BOOTSTRAP_PERIOD = os.getenv("BOOTSTRAP_PERIOD", "730d")
TICKER = os.getenv("BOOTSTRAP_TICKER", "BTC-USD")

HOLD_HOURS = [12, 24, 48, 72, 96]
STOP_MULTS = [1.0, 1.5, 2.0, 3.0]
MIN_STOP_PCT = 0.0075

# Chop / trend-clean research tags. These do not block trades yet; they are logged for later analysis.
CHOP_LOOKBACK = int(os.getenv("CHOP_LOOKBACK", "24"))
CHOP_MEDIAN_LOOKBACK = int(os.getenv("CHOP_MEDIAN_LOOKBACK", "96"))
CHOP_TREND_MAX = float(os.getenv("CHOP_TREND_MAX", "34"))
CHOP_CHOP_MIN = float(os.getenv("CHOP_CHOP_MIN", "60"))

FRESH_CONTEXT_BARS = 2
ESTABLISHED_MAX_CONTEXT_BARS = 6

MODEL_SPECS = [
    {
        "model": "BTC_SHORT_A_1H_8H_BREAKOUT_EST_BEAR",
        "label": "Short A: 1h/8h established BEAR breakdown",
        "exec_tf": "1h",
        "context_tf": "8h",
        "family": "breakout_continuation",
        "side": "SHORT",
        "age_bucket": "established",
    },
    {
        "model": "BTC_SHORT_B_2H_8H_BREAKOUT_EST_BEAR",
        "label": "Short B: 2h/8h established BEAR breakdown",
        "exec_tf": "2h",
        "context_tf": "8h",
        "family": "breakout_continuation",
        "side": "SHORT",
        "age_bucket": "established",
    },
    {
        "model": "BTC_LONG_C_1H_4H_PULLBACK_EST_BULL",
        "label": "Long C: 1h/4h established BULL pullback reclaim",
        "exec_tf": "1h",
        "context_tf": "4h",
        "family": "pullback_reclaim",
        "side": "LONG",
        "age_bucket": "established",
    },
]

app = Flask(__name__)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value):
    if value is None or value == "":
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 10_000_000_000:
            v = v / 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc).replace(second=0, microsecond=0)
    s = str(value).strip()
    try:
        if s.isdigit():
            v = float(s)
            if v > 10_000_000_000:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc).replace(second=0, microsecond=0)
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc).replace(second=0, microsecond=0)
    except Exception:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def iso(dt):
    if isinstance(dt, str):
        return dt
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def fnum(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_columns(conn, table, columns):
    """Small SQLite migration helper for existing Railway volumes."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_db():
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles_1h (
                timestamp TEXT PRIMARY KEY,
                symbol TEXT,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                source TEXT,
                received_at TEXT,
                chop_score_1h REAL,
                chop_state_1h TEXT,
                directional_efficiency_24h REAL,
                range_compression_24h REAL,
                ema_flatness_pct REAL,
                atr_compression_ratio REAL,
                failed_breakout_flag INTEGER DEFAULT 0,
                trend_clean_flag INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                model TEXT,
                label TEXT,
                side TEXT,
                exec_tf TEXT,
                context_tf TEXT,
                family TEXT,
                signal_time TEXT,
                entry_time TEXT,
                entry_price REAL,
                entry_atr_pct REAL,
                signal_close REAL,
                ctx_regime TEXT,
                ctx_age REAL,
                age_bucket TEXT,
                ctx_bull_score REAL,
                ctx_bear_score REAL,
                ctx_far_now INTEGER,
                chop_score REAL,
                chop_state TEXT,
                directional_efficiency_24h REAL,
                range_compression_24h REAL,
                ema_flatness_pct REAL,
                atr_compression_ratio REAL,
                failed_breakout_flag INTEGER DEFAULT 0,
                trend_clean_flag INTEGER DEFAULT 0,
                current_price REAL,
                current_return_pct REAL,
                mfe_pct REAL,
                mae_pct REAL,
                reached_1r_2atr INTEGER,
                hours_to_1r_2atr REAL,
                stop_1atr_hit INTEGER DEFAULT 0,
                stop_15atr_hit INTEGER DEFAULT 0,
                stop_2atr_hit INTEGER DEFAULT 0,
                stop_3atr_hit INTEGER DEFAULT 0,
                stop_1atr_hit_time TEXT,
                stop_15atr_hit_time TEXT,
                stop_2atr_hit_time TEXT,
                stop_3atr_hit_time TEXT,
                ret_12h_pct REAL,
                ret_24h_pct REAL,
                ret_48h_pct REAL,
                ret_72h_pct REAL,
                ret_96h_pct REAL,
                status TEXT DEFAULT 'OPEN',
                updated_at TEXT,
                UNIQUE(model, signal_time)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                event_type TEXT,
                message TEXT,
                payload TEXT
            )
            """
        )
        ensure_columns(conn, "candles_1h", {
            "chop_score_1h": "REAL",
            "chop_state_1h": "TEXT",
            "directional_efficiency_24h": "REAL",
            "range_compression_24h": "REAL",
            "ema_flatness_pct": "REAL",
            "atr_compression_ratio": "REAL",
            "failed_breakout_flag": "INTEGER DEFAULT 0",
            "trend_clean_flag": "INTEGER DEFAULT 0",
        })
        ensure_columns(conn, "shadow_trades", {
            "chop_score": "REAL",
            "chop_state": "TEXT",
            "directional_efficiency_24h": "REAL",
            "range_compression_24h": "REAL",
            "ema_flatness_pct": "REAL",
            "atr_compression_ratio": "REAL",
            "failed_breakout_flag": "INTEGER DEFAULT 0",
            "trend_clean_flag": "INTEGER DEFAULT 0",
        })
        conn.commit()


def log_event(event_type, message, payload=None):
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO events(created_at,event_type,message,payload) VALUES(?,?,?,?)",
                (now_iso(), event_type, message, json.dumps(payload or {}, default=str)),
            )
            conn.commit()
    except Exception:
        pass


def upsert_candle(ts, symbol, o, h, l, c, v=None, source="webhook"):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO candles_1h(timestamp,symbol,open,high,low,close,volume,source,received_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(timestamp) DO UPDATE SET
                symbol=excluded.symbol,
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                source=excluded.source,
                received_at=excluded.received_at
            """,
            (iso(ts), symbol, o, h, l, c, v, source, now_iso()),
        )
        conn.commit()


def load_candles_df():
    with connect() as conn:
        df = pd.read_sql_query("SELECT * FROM candles_1h ORDER BY timestamp", conn)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])


def tf_to_rule(tf):
    tf = tf.lower().strip()
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    if tf.endswith("d"):
        return f"{int(tf[:-1])}D"
    raise ValueError(f"Unsupported timeframe: {tf}")


def tf_to_hours(tf):
    tf = tf.lower().strip()
    if tf.endswith("h"):
        return int(tf[:-1])
    if tf.endswith("d"):
        return int(tf[:-1]) * 24
    raise ValueError(f"Unsupported timeframe: {tf}")


def resample_ohlcv(df, tf):
    return df.resample(tf_to_rule(tf), label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_exec_indicators(df):
    d = df.copy()
    d["ema8"] = ema(d["close"], 8)
    d["ema21"] = ema(d["close"], 21)
    d["ema55"] = ema(d["close"], 55)
    d["ema144"] = ema(d["close"], 144)
    d["rsi14"] = rsi(d["close"], 14)
    d["atr14"] = atr(d, 14)
    d["atr_pct"] = d["atr14"] / d["close"]
    d["roll_high20"] = d["high"].shift(1).rolling(20).max()
    d["roll_low20"] = d["low"].shift(1).rolling(20).min()

    # Chop research labels. These are deliberately broad, not an entry filter yet.
    # The aim is to later compare candidate performance in TREND vs MIXED vs CHOP.
    lookback = max(6, CHOP_LOOKBACK)
    med_lookback = max(lookback * 2, CHOP_MEDIAN_LOOKBACK)

    rolling_high = d["high"].rolling(lookback).max()
    rolling_low = d["low"].rolling(lookback).min()
    range_abs = (rolling_high - rolling_low).replace(0, np.nan)
    d["directional_efficiency_24h"] = (d["close"] - d["close"].shift(lookback)).abs() / range_abs

    range_pct = range_abs / d["close"]
    range_pct_med = range_pct.rolling(med_lookback).median().replace(0, np.nan)
    d["range_compression_24h"] = range_pct / range_pct_med

    d["ema_flatness_pct"] = (d["ema21"] - d["ema55"]).abs() / d["close"]
    atr_med = d["atr_pct"].rolling(med_lookback).median().replace(0, np.nan)
    d["atr_compression_ratio"] = d["atr_pct"] / atr_med

    candle_dir = np.sign(d["close"] - d["open"])
    d["alternating_rate_12h"] = (candle_dir != candle_dir.shift(1)).rolling(12).mean()

    failed_up = (d["high"] > d["roll_high20"]) & (d["close"] < d["roll_high20"])
    failed_down = (d["low"] < d["roll_low20"]) & (d["close"] > d["roll_low20"])
    d["failed_breakout_flag"] = (failed_up | failed_down).fillna(False).astype(int)

    ema_flat_threshold = np.maximum(0.006, d["atr_pct"].fillna(0) * 0.80)
    score = pd.Series(0.0, index=d.index)
    score += np.where(d["directional_efficiency_24h"] < 0.25, 30.0, 0.0)
    score += np.where(d["range_compression_24h"] < 0.75, 20.0, 0.0)
    score += np.where(d["ema_flatness_pct"] < ema_flat_threshold, 20.0, 0.0)
    score += np.where(d["atr_compression_ratio"] < 0.80, 15.0, 0.0)
    score += np.where(d["alternating_rate_12h"] > 0.55, 15.0, 0.0)
    score += np.where(d["failed_breakout_flag"] == 1, 10.0, 0.0)

    # Warmup rows are uncertain; keep them as MIXED rather than falsely TREND/CHOP.
    warmup_mask = d[["directional_efficiency_24h", "range_compression_24h", "atr_compression_ratio"]].isna().any(axis=1)
    score = score.clip(lower=0.0, upper=100.0)
    score = score.mask(warmup_mask, 50.0)
    d["chop_score"] = score
    d["chop_state"] = np.select(
        [d["chop_score"] >= CHOP_CHOP_MIN, d["chop_score"] <= CHOP_TREND_MAX],
        ["CHOP", "TREND"],
        default="MIXED",
    )
    d["trend_clean_flag"] = (d["chop_state"] == "TREND").astype(int)
    return d

def rolling_regime_age(regime, target):
    ages = []
    age = np.nan
    for val in regime.astype(str).values:
        if val == target:
            age = 0 if pd.isna(age) else age + 1
        else:
            age = np.nan
        ages.append(age)
    return pd.Series(ages, index=regime.index)


def add_context_indicators(df):
    d = add_exec_indicators(df)
    macd = ema(d["close"], 12) - ema(d["close"], 26)
    macd_sig = ema(macd, 9)
    bull_terms = [
        d["close"] > d["ema8"],
        d["close"] > d["ema21"],
        d["close"] > d["ema55"],
        d["close"] > d["ema144"],
        d["ema8"] > d["ema21"],
        d["ema21"] > d["ema55"],
        d["ema55"] > d["ema144"],
        d["ema21"] > d["ema21"].shift(3),
        d["rsi14"] > 50,
        macd > macd_sig,
    ]
    bear_terms = [
        d["close"] < d["ema8"],
        d["close"] < d["ema21"],
        d["close"] < d["ema55"],
        d["close"] < d["ema144"],
        d["ema8"] < d["ema21"],
        d["ema21"] < d["ema55"],
        d["ema55"] < d["ema144"],
        d["ema21"] < d["ema21"].shift(3),
        d["rsi14"] < 50,
        macd < macd_sig,
    ]
    d["ctx_bull_score"] = sum(x.astype(int) for x in bull_terms)
    d["ctx_bear_score"] = sum(x.astype(int) for x in bear_terms)
    d["ctx_bull_stack"] = (d["ema8"] > d["ema21"]) & (d["ema21"] > d["ema55"]) & (d["ema55"] > d["ema144"])
    d["ctx_bear_stack"] = (d["ema8"] < d["ema21"]) & (d["ema21"] < d["ema55"]) & (d["ema55"] < d["ema144"])
    d["ctx_dist_ema21_pct"] = (d["close"] - d["ema21"]) / d["close"]
    d["ctx_atr_pct"] = d["atr_pct"]
    stretch = np.maximum(2.0 * d["ctx_atr_pct"].fillna(0), 0.04)
    d["ctx_far_up"] = d["ctx_dist_ema21_pct"] > stretch
    d["ctx_far_down"] = d["ctx_dist_ema21_pct"] < -stretch
    d["ctx_regime"] = np.select(
        [(d["ctx_bull_score"] >= 7) & d["ctx_bull_stack"], (d["ctx_bear_score"] >= 7) & d["ctx_bear_stack"]],
        ["BULL", "BEAR"],
        default="NEUTRAL",
    )
    regime = pd.Series(d["ctx_regime"], index=d.index)
    d["ctx_bull_age"] = rolling_regime_age(regime, "BULL")
    d["ctx_bear_age"] = rolling_regime_age(regime, "BEAR")
    return d[
        [
            "ctx_bull_score", "ctx_bear_score", "ctx_regime", "ctx_bull_age", "ctx_bear_age",
            "ctx_far_up", "ctx_far_down", "ctx_atr_pct", "ctx_dist_ema21_pct"
        ]
    ]


def merge_context(exec_df, ctx_df):
    e = exec_df.reset_index().rename(columns={exec_df.index.name or "index": "timestamp"})
    if "timestamp" not in e.columns:
        e = e.rename(columns={e.columns[0]: "timestamp"})
    c = ctx_df.reset_index().rename(columns={ctx_df.index.name or "index": "timestamp"})
    if "timestamp" not in c.columns:
        c = c.rename(columns={c.columns[0]: "timestamp"})
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
    out = pd.merge_asof(e.sort_values("timestamp"), c.sort_values("timestamp"), on="timestamp", direction="backward")
    return out.set_index("timestamp").sort_index()


def age_bucket(age):
    if pd.isna(age):
        return "none"
    if age <= FRESH_CONTEXT_BARS:
        return "fresh"
    if age <= ESTABLISHED_MAX_CONTEXT_BARS:
        return "established"
    return "mature"


def model_candidate(row, spec):
    side = spec["side"]
    family = spec["family"]
    if side == "SHORT":
        ctx_age = row.get("ctx_bear_age", np.nan)
        far_now = bool(row.get("ctx_far_down", False))
        base_env = row.get("ctx_regime") == "BEAR"
    else:
        ctx_age = row.get("ctx_bull_age", np.nan)
        far_now = bool(row.get("ctx_far_up", False))
        base_env = row.get("ctx_regime") == "BULL"

    bucket = age_bucket(ctx_age)
    if not base_env or bucket != spec["age_bucket"]:
        return False, ctx_age, bucket, far_now

    if family == "breakout_continuation" and side == "SHORT":
        ok = (row["close"] < row["roll_low20"]) and (row["close"] < row["ema21"]) and (row["rsi14"] <= 45)
    elif family == "pullback_reclaim" and side == "LONG":
        ok = (row["low"] <= row["ema21"]) and (row["close"] > row["ema8"]) and (row["close"] > row["open"]) and (row["close"] > row["ema55"]) and (row["rsi14"] >= 45)
    else:
        ok = False
    return bool(ok), ctx_age, bucket, far_now


def build_model_frame(base_df, exec_tf, context_tf):
    exec_bars = add_exec_indicators(resample_ohlcv(base_df, exec_tf)).dropna()
    ctx_feat = add_context_indicators(resample_ohlcv(base_df, context_tf))
    merged = merge_context(exec_bars, ctx_feat).dropna(subset=["ema8", "ema21", "ema55", "atr_pct", "ctx_regime"])
    return merged


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def save_latest_chop_metrics(base_df):
    """Persist the latest 1h chop tag onto candles_1h for export/dashboard analysis."""
    try:
        frame = add_exec_indicators(base_df).dropna(subset=["close"])
        if frame.empty:
            return None
        ts = frame.index.max()
        row = frame.loc[ts]
        payload = {
            "chop_score_1h": safe_float(row.get("chop_score")),
            "chop_state_1h": str(row.get("chop_state")),
            "directional_efficiency_24h": safe_float(row.get("directional_efficiency_24h")),
            "range_compression_24h": safe_float(row.get("range_compression_24h")),
            "ema_flatness_pct": safe_float(row.get("ema_flatness_pct")),
            "atr_compression_ratio": safe_float(row.get("atr_compression_ratio")),
            "failed_breakout_flag": int(row.get("failed_breakout_flag", 0) or 0),
            "trend_clean_flag": int(row.get("trend_clean_flag", 0) or 0),
        }
        with connect() as conn:
            conn.execute(
                """
                UPDATE candles_1h
                SET chop_score_1h=?, chop_state_1h=?, directional_efficiency_24h=?, range_compression_24h=?,
                    ema_flatness_pct=?, atr_compression_ratio=?, failed_breakout_flag=?, trend_clean_flag=?
                WHERE timestamp=?
                """,
                (
                    payload["chop_score_1h"], payload["chop_state_1h"], payload["directional_efficiency_24h"],
                    payload["range_compression_24h"], payload["ema_flatness_pct"], payload["atr_compression_ratio"],
                    payload["failed_breakout_flag"], payload["trend_clean_flag"], iso(ts),
                ),
            )
            conn.commit()
        return payload
    except Exception as e:
        log_event("chop_update_error", str(e), {})
        return None


def refresh_all_chop_metrics():
    """Backfill chop tags onto all stored 1h candles, useful after historical bootstrap."""
    try:
        base_df = load_candles_df()
        if base_df.empty:
            return {"ok": True, "rows": 0}
        frame = add_exec_indicators(base_df).dropna(subset=["close"])
        rows = []
        for ts, row in frame.iterrows():
            rows.append((
                safe_float(row.get("chop_score")), str(row.get("chop_state")),
                safe_float(row.get("directional_efficiency_24h")), safe_float(row.get("range_compression_24h")),
                safe_float(row.get("ema_flatness_pct")), safe_float(row.get("atr_compression_ratio")),
                int(row.get("failed_breakout_flag", 0) or 0), int(row.get("trend_clean_flag", 0) or 0), iso(ts),
            ))
        with connect() as conn:
            conn.executemany(
                """
                UPDATE candles_1h
                SET chop_score_1h=?, chop_state_1h=?, directional_efficiency_24h=?, range_compression_24h=?,
                    ema_flatness_pct=?, atr_compression_ratio=?, failed_breakout_flag=?, trend_clean_flag=?
                WHERE timestamp=?
                """,
                rows,
            )
            conn.commit()
        return {"ok": True, "rows": len(rows)}
    except Exception as e:
        log_event("chop_backfill_error", str(e), {})
        return {"ok": False, "error": str(e)}


def process_latest_candle():
    base = load_candles_df()
    if len(base) < 300:
        return {"ok": False, "message": f"Need more warmup candles. Have {len(base)}."}

    latest_ts = base.index.max()
    latest_chop = save_latest_chop_metrics(base)
    created = []
    states = []

    for spec in MODEL_SPECS:
        try:
            frame = build_model_frame(base, spec["exec_tf"], spec["context_tf"])
            if frame.empty:
                continue
            latest_exec_ts = frame.index.max()
            if latest_exec_ts != latest_ts:
                states.append({"model": spec["model"], "candidate": False, "reason": "waiting_for_exec_close", "latest_exec_ts": str(latest_exec_ts), "latest_ts": str(latest_ts)})
                continue
            row = frame.iloc[-1]
            ok, ctx_age, bucket, far_now = model_candidate(row, spec)
            states.append({
                "model": spec["model"], "candidate": ok, "side": spec["side"], "ctx_regime": row.get("ctx_regime"),
                "ctx_age": None if pd.isna(ctx_age) else float(ctx_age), "age_bucket": bucket,
                "ctx_bull_score": float(row.get("ctx_bull_score", np.nan)), "ctx_bear_score": float(row.get("ctx_bear_score", np.nan)),
                "ctx_far_now": far_now, "close": float(row["close"]), "time": str(latest_exec_ts),
                "chop_score": safe_float(row.get("chop_score")), "chop_state": str(row.get("chop_state")),
                "trend_clean_flag": int(row.get("trend_clean_flag", 0) or 0),
                "failed_breakout_flag": int(row.get("failed_breakout_flag", 0) or 0)
            })
            if ok:
                trade_id = create_shadow_trade(spec, latest_exec_ts, row, ctx_age, bucket, far_now)
                if trade_id:
                    created.append({"id": trade_id, "model": spec["model"], "side": spec["side"], "time": str(latest_exec_ts)})
        except Exception as e:
            log_event("model_error", f"{spec['model']}: {e}", {"spec": spec})
            states.append({"model": spec["model"], "candidate": False, "error": str(e)})

    update_all_trades()
    return {"ok": True, "latest_ts": str(latest_ts), "latest_chop": latest_chop, "created": created, "states": states}


def create_shadow_trade(spec, signal_time, row, ctx_age, bucket, far_now):
    entry_price = float(row["close"])
    entry_atr_pct = float(row["atr_pct"])
    with connect() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO shadow_trades(
                    created_at,model,label,side,exec_tf,context_tf,family,signal_time,entry_time,entry_price,entry_atr_pct,
                    signal_close,ctx_regime,ctx_age,age_bucket,ctx_bull_score,ctx_bear_score,ctx_far_now,
                    chop_score,chop_state,directional_efficiency_24h,range_compression_24h,ema_flatness_pct,atr_compression_ratio,
                    failed_breakout_flag,trend_clean_flag,current_price,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now_iso(), spec["model"], spec["label"], spec["side"], spec["exec_tf"], spec["context_tf"], spec["family"],
                    iso(signal_time), iso(signal_time), entry_price, entry_atr_pct, entry_price, row.get("ctx_regime"),
                    None if pd.isna(ctx_age) else float(ctx_age), bucket, float(row.get("ctx_bull_score", np.nan)),
                    float(row.get("ctx_bear_score", np.nan)), 1 if far_now else 0,
                    safe_float(row.get("chop_score")), str(row.get("chop_state")),
                    safe_float(row.get("directional_efficiency_24h")), safe_float(row.get("range_compression_24h")),
                    safe_float(row.get("ema_flatness_pct")), safe_float(row.get("atr_compression_ratio")),
                    int(row.get("failed_breakout_flag", 0) or 0), int(row.get("trend_clean_flag", 0) or 0),
                    entry_price, now_iso(),
                ),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None


def trade_return_pct(side, entry, price):
    return ((price - entry) / entry * 100.0) if side == "LONG" else ((entry - price) / entry * 100.0)


def stop_price(side, entry, risk_pct):
    return entry * (1 - risk_pct) if side == "LONG" else entry * (1 + risk_pct)


def update_all_trades():
    candles = load_candles_df()
    if candles.empty:
        return
    latest_close = float(candles["close"].iloc[-1])
    latest_ts = candles.index.max()
    with connect() as conn:
        trades = conn.execute("SELECT * FROM shadow_trades WHERE status != 'ARCHIVED' ORDER BY entry_time").fetchall()
        for t in trades:
            entry_time = pd.to_datetime(t["entry_time"], utc=True)
            entry = float(t["entry_price"])
            side = t["side"]
            entry_atr_pct = float(t["entry_atr_pct"] or 0.0)
            window = candles[candles.index > entry_time]
            if window.empty:
                continue
            current = latest_close
            current_ret = trade_return_pct(side, entry, current)
            if side == "LONG":
                mfe_pct = (float(window["high"].max()) - entry) / entry * 100.0
                mae_pct = (float(window["low"].min()) - entry) / entry * 100.0
            else:
                mfe_pct = (entry - float(window["low"].min())) / entry * 100.0
                mae_pct = (entry - float(window["high"].max())) / entry * 100.0

            updates = {
                "current_price": current,
                "current_return_pct": current_ret,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "updated_at": now_iso(),
                "status": "COMPLETE" if latest_ts >= entry_time + pd.Timedelta(hours=96) else "OPEN",
            }

            risk_2 = max(2.0 * entry_atr_pct, MIN_STOP_PCT)
            if not t["reached_1r_2atr"]:
                if side == "LONG":
                    hit = window[window["high"] >= entry * (1 + risk_2)]
                else:
                    hit = window[window["low"] <= entry * (1 - risk_2)]
                if not hit.empty:
                    first = hit.index.min()
                    updates["reached_1r_2atr"] = 1
                    updates["hours_to_1r_2atr"] = (first - entry_time).total_seconds() / 3600.0

            for mult, col, time_col in [
                (1.0, "stop_1atr_hit", "stop_1atr_hit_time"),
                (1.5, "stop_15atr_hit", "stop_15atr_hit_time"),
                (2.0, "stop_2atr_hit", "stop_2atr_hit_time"),
                (3.0, "stop_3atr_hit", "stop_3atr_hit_time"),
            ]:
                if not t[col]:
                    risk = max(mult * entry_atr_pct, MIN_STOP_PCT)
                    sp = stop_price(side, entry, risk)
                    if side == "LONG":
                        hit = window[window["low"] <= sp]
                    else:
                        hit = window[window["high"] >= sp]
                    if not hit.empty:
                        updates[col] = 1
                        updates[time_col] = iso(hit.index.min())

            for h in HOLD_HOURS:
                col = f"ret_{h}h_pct"
                if t[col] is None and latest_ts >= entry_time + pd.Timedelta(hours=h):
                    target = entry_time + pd.Timedelta(hours=h)
                    exit_rows = candles[candles.index <= target]
                    if not exit_rows.empty:
                        exit_price = float(exit_rows["close"].iloc[-1])
                        updates[col] = trade_return_pct(side, entry, exit_price)

            set_clause = ", ".join([f"{k}=?" for k in updates.keys()])
            vals = list(updates.values()) + [t["id"]]
            conn.execute(f"UPDATE shadow_trades SET {set_clause} WHERE id=?", vals)
        conn.commit()


def _normalise_yfinance_ohlcv(raw):
    """Return a clean OHLCV dataframe from yfinance output, or empty dataframe."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna()
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _period_days(period):
    """Convert simple period strings like 730d, 365d, 24mo into approximate days."""
    s = str(period or "730d").strip().lower()
    try:
        if s.endswith("d"):
            return max(1, int(float(s[:-1])))
        if s.endswith("mo"):
            return max(1, int(float(s[:-2]) * 30))
        if s.endswith("y"):
            return max(1, int(float(s[:-1]) * 365))
        return max(1, int(float(s)))
    except Exception:
        return 730


def _period_bounds_utc(period):
    days = _period_days(period)
    now = datetime.now(timezone.utc)
    # Use the last fully closed hourly candle, not the currently-forming one.
    end = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start = end - timedelta(days=days)
    return start, end


def _http_json(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 BTC-Regime-Research-Logger/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _yfinance_period_candidates(period):
    """Yahoo can occasionally reject/return empty for 730d. Try safer fallbacks."""
    candidates = []
    for x in [period, "729d", "720d", "365d", "180d", "90d", "60d", "30d"]:
        if x and x not in candidates:
            candidates.append(x)
    return candidates


def _download_btc_yfinance(ticker, period):
    attempts = []
    if yf is None:
        return pd.DataFrame(), [{"method": "yfinance", "error": "yfinance not installed"}], None, None
    for p in _yfinance_period_candidates(period):
        # Method 1: yf.download. Keep threads off on Railway; it is more predictable.
        try:
            raw = yf.download(
                ticker,
                interval="1h",
                period=p,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _normalise_yfinance_ohlcv(raw)
            attempts.append({"method": "yf.download", "ticker": ticker, "period": p, "rows": int(len(df))})
            if not df.empty:
                return df, attempts, p, "yf.download"
        except Exception as e:
            attempts.append({"method": "yf.download", "ticker": ticker, "period": p, "error": str(e)[:250]})

        # Method 2: Ticker.history fallback.
        try:
            raw = yf.Ticker(ticker).history(
                interval="1h",
                period=p,
                auto_adjust=False,
                actions=False,
            )
            df = _normalise_yfinance_ohlcv(raw)
            attempts.append({"method": "Ticker.history", "ticker": ticker, "period": p, "rows": int(len(df))})
            if not df.empty:
                return df, attempts, p, "Ticker.history"
        except Exception as e:
            attempts.append({"method": "Ticker.history", "ticker": ticker, "period": p, "error": str(e)[:250]})

    return pd.DataFrame(), attempts, None, None


def _download_btc_binance(period, symbol=None):
    """Bootstrap BTCUSDT hourly candles from Binance public klines. No API key."""
    symbol = (symbol or os.getenv("BOOTSTRAP_BINANCE_SYMBOL", "BTCUSDT")).upper().strip()
    start, end = _period_bounds_utc(period)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((end + timedelta(minutes=59, seconds=59)).timestamp() * 1000)
    attempts = []
    rows = []
    cursor = start_ms
    limit = 1000

    while cursor <= end_ms:
        params = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": "1h",
            "limit": limit,
            "startTime": cursor,
            "endTime": end_ms,
        })
        url = f"https://api.binance.com/api/v3/klines?{params}"
        try:
            data = _http_json(url, timeout=25)
            attempts.append({"method": "binance.klines", "symbol": symbol, "start_ms": cursor, "rows": len(data) if isinstance(data, list) else 0})
        except Exception as e:
            attempts.append({"method": "binance.klines", "symbol": symbol, "start_ms": cursor, "error": str(e)[:250]})
            break

        if not isinstance(data, list) or not data:
            break

        rows.extend(data)
        last_open_ms = int(data[-1][0])
        next_cursor = last_open_ms + 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(data) < limit:
            break

    if not rows:
        return pd.DataFrame(), attempts, symbol, "binance.klines"

    out = []
    for k in rows:
        # Binance kline: open time, open, high, low, close, volume, close time, ...
        out.append({
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(out).dropna()
    if df.empty:
        return pd.DataFrame(), attempts, symbol, "binance.klines"
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]], attempts, symbol, "binance.klines"


def _download_btc_coinbase(period, product=None):
    """Bootstrap BTC-USD hourly candles from Coinbase Exchange public candles. No API key."""
    product = (product or os.getenv("BOOTSTRAP_COINBASE_PRODUCT", "BTC-USD")).upper().strip()
    start, end = _period_bounds_utc(period)
    attempts = []
    rows = []
    chunk = timedelta(hours=299)  # Coinbase max is around 300 candles.
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        params = urllib.parse.urlencode({
            "granularity": 3600,
            "start": cursor.isoformat().replace("+00:00", "Z"),
            "end": chunk_end.isoformat().replace("+00:00", "Z"),
        })
        url = f"https://api.exchange.coinbase.com/products/{urllib.parse.quote(product)}/candles?{params}"
        try:
            data = _http_json(url, timeout=25)
            attempts.append({"method": "coinbase.candles", "product": product, "start": cursor.isoformat(), "rows": len(data) if isinstance(data, list) else 0})
        except Exception as e:
            attempts.append({"method": "coinbase.candles", "product": product, "start": cursor.isoformat(), "error": str(e)[:250]})
            break

        if isinstance(data, list) and data:
            rows.extend(data)
        cursor = chunk_end + timedelta(hours=1)

    if not rows:
        return pd.DataFrame(), attempts, product, "coinbase.candles"

    out = []
    for c in rows:
        # Coinbase candle: [time, low, high, open, close, volume]
        if len(c) < 6:
            continue
        out.append({
            "timestamp": pd.to_datetime(int(c[0]), unit="s", utc=True),
            "open": float(c[3]),
            "high": float(c[2]),
            "low": float(c[1]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })
    df = pd.DataFrame(out).dropna()
    if df.empty:
        return pd.DataFrame(), attempts, product, "coinbase.candles"
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]], attempts, product, "coinbase.candles"


def bootstrap_yfinance(force=False, ticker=None, period=None):
    """Historical candle bootstrap. Name kept for compatibility; now tries Yahoo, Binance and Coinbase."""
    ticker = ticker or TICKER
    period = period or BOOTSTRAP_PERIOD
    all_attempts = []

    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM candles_1h").fetchone()["n"]
    if count > 0 and not force:
        return {"ok": True, "version": APP_VERSION, "message": f"Skipped bootstrap; {count} candles already in DB", "candles": count}

    # 1) Try Yahoo first because it matches the Colab source.
    df, attempts, used_period, used_method = _download_btc_yfinance(ticker, period)
    all_attempts.extend(attempts)
    used_source = "yfinance"
    used_symbol = ticker

    # 2) Railway sometimes gets empty Yahoo data. Binance hourly klines are a robust no-key fallback.
    if df.empty:
        df, attempts, used_symbol, used_method = _download_btc_binance(period)
        all_attempts.extend(attempts)
        used_period = period
        used_source = "binance"

    # 3) Coinbase fallback if Binance is blocked/unavailable.
    if df.empty:
        df, attempts, used_symbol, used_method = _download_btc_coinbase(period)
        all_attempts.extend(attempts)
        used_period = period
        used_source = "coinbase"

    if df.empty:
        log_event("bootstrap_error", "No historical data from Yahoo, Binance or Coinbase", {"ticker": ticker, "period": period, "attempts": all_attempts})
        return {
            "ok": False,
            "version": APP_VERSION,
            "message": "No historical data from Yahoo, Binance or Coinbase",
            "ticker": ticker,
            "period": period,
            "attempts": all_attempts[-20:],
        }

    rows = 0
    for ts, r in df.iterrows():
        upsert_candle(ts, used_symbol, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume), source=used_source)
        rows += 1
    chop_result = refresh_all_chop_metrics()
    log_event("bootstrap", f"Bootstrapped {rows} candles", {"source": used_source, "symbol": used_symbol, "period": used_period, "method": used_method, "chop_backfill": chop_result})
    return {
        "ok": True,
        "version": APP_VERSION,
        "message": f"Bootstrapped {rows} candles",
        "candles": rows,
        "source": used_source,
        "symbol": used_symbol,
        "period": used_period,
        "method": used_method,
        "attempts": all_attempts[-20:],
        "chop_backfill": chop_result,
    }


def table_df(table):
    with connect() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


@app.route("/health")
def health():
    candles = table_df("candles_1h")
    trades = table_df("shadow_trades")
    return jsonify({
        "ok": True,
        "app": APP_NAME,
        "version": APP_VERSION,
        "db_path": DB_PATH,
        "bootstrap_ticker": TICKER,
        "bootstrap_period": BOOTSTRAP_PERIOD,
        "candles": int(len(candles)),
        "trades": int(len(trades)),
        "latest_candle": None if candles.empty else candles["timestamp"].max(),
        "models": [m["model"] for m in MODEL_SPECS],
    })


@app.route("/bootstrap")
def bootstrap_route():
    force = request.args.get("force", "false").lower() in ("1", "true", "yes")
    ticker = request.args.get("ticker") or TICKER
    period = request.args.get("period") or BOOTSTRAP_PERIOD
    result = bootstrap_yfinance(force=force, ticker=ticker, period=period)
    return jsonify(result)


@app.route("/refresh_chop")
def refresh_chop_route():
    return jsonify(refresh_all_chop_metrics())


@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.get_json(silent=True) or {}

    # TradingView cannot easily send custom HTTP headers, so the Pine script sends
    # the shared secret inside the JSON alert body as "secret". We also accept
    # common alternate body/header names for manual tests or future changes.
    secret = (
        payload.get("secret")
        or payload.get("webhook_secret")
        or payload.get("WEBHOOK_SECRET")
        or request.headers.get("X-Webhook-Secret")
        or request.headers.get("X-Webhook-Secret".lower())
    )

    if WEBHOOK_SECRET and WEBHOOK_SECRET != "change-me" and secret != WEBHOOK_SECRET:
        log_event("rejected_webhook", "bad secret", {"received_secret_present": bool(secret), "payload_keys": list(payload.keys())})
        return jsonify({"ok": False, "error": "bad secret"}), 403

    ts = parse_timestamp(payload.get("time_close") or payload.get("timestamp") or payload.get("time"))
    symbol = str(payload.get("symbol") or payload.get("ticker") or "BTC").upper()
    o = fnum(payload.get("open"))
    h = fnum(payload.get("high"))
    l = fnum(payload.get("low"))
    c = fnum(payload.get("close"))
    v = fnum(payload.get("volume"), 0.0)

    if None in (o, h, l, c):
        log_event("bad_webhook", "missing OHLC", payload)
        return jsonify({"ok": False, "error": "missing OHLC", "payload": payload}), 400

    upsert_candle(ts, symbol, o, h, l, c, v, source="tradingview")
    result = process_latest_candle()
    log_event("webhook", "processed candle", {"timestamp": iso(ts), "symbol": symbol, "close": c, "result": result})
    return jsonify(result)


@app.route("/export")
def export():
    update_all_trades()
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for table in ["candles_1h", "shadow_trades", "events"]:
            df = table_df(table)
            z.writestr(f"{table}.csv", df.to_csv(index=False))
    mem.seek(0)
    return Response(
        mem.read(),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=btc_regime_research_export.zip"},
    )


def fmt(x, nd=2):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def dashboard_data():
    update_all_trades()
    candles = table_df("candles_1h")
    trades = table_df("shadow_trades")
    events = table_df("events")
    latest = None if candles.empty else candles.sort_values("timestamp").iloc[-1].to_dict()
    if not trades.empty:
        for col in [
            "current_return_pct", "ret_24h_pct", "ret_48h_pct", "ret_72h_pct", "ret_96h_pct",
            "mfe_pct", "mae_pct", "chop_score", "stop_2atr_hit", "reached_1r_2atr", "trend_clean_flag",
        ]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")
        summary = trades.groupby(["model", "side"]).agg(
            trades=("id", "count"),
            open=("status", lambda x: int((x == "OPEN").sum())),
            avg_24h=("ret_24h_pct", "mean"),
            avg_48h=("ret_48h_pct", "mean"),
            avg_96h=("ret_96h_pct", "mean"),
            stop_2atr_rate=("stop_2atr_hit", lambda x: 100.0 * pd.to_numeric(x, errors="coerce").fillna(0).mean()),
            reached_1r_rate=("reached_1r_2atr", lambda x: 100.0 * pd.to_numeric(x, errors="coerce").fillna(0).mean()),
        ).reset_index()
        if "chop_state" in trades.columns:
            chop_summary = trades.groupby(["model", "side", "chop_state"], dropna=False).agg(
                trades=("id", "count"),
                open=("status", lambda x: int((x == "OPEN").sum())),
                avg_chop_score=("chop_score", "mean"),
                avg_24h=("ret_24h_pct", "mean"),
                avg_48h=("ret_48h_pct", "mean"),
                avg_96h=("ret_96h_pct", "mean"),
                stop_2atr_rate=("stop_2atr_hit", lambda x: 100.0 * pd.to_numeric(x, errors="coerce").fillna(0).mean()),
                reached_1r_rate=("reached_1r_2atr", lambda x: 100.0 * pd.to_numeric(x, errors="coerce").fillna(0).mean()),
            ).reset_index()
        else:
            chop_summary = pd.DataFrame()
        latest_trades = trades.sort_values("id", ascending=False).head(30)
    else:
        summary = pd.DataFrame()
        chop_summary = pd.DataFrame()
        latest_trades = pd.DataFrame()
    latest_events = events.sort_values("id", ascending=False).head(20) if not events.empty else pd.DataFrame()
    return latest, trades, summary, chop_summary, latest_trades, latest_events

@app.route("/")
def index():
    latest, trades, summary, chop_summary, latest_trades, latest_events = dashboard_data()
    total_trades = 0 if trades.empty else len(trades)
    open_trades = 0 if trades.empty else int((trades["status"] == "OPEN").sum())
    completed_96 = 0 if trades.empty else int(trades["ret_96h_pct"].notna().sum())
    latest_price = "—" if latest is None else fmt(latest.get("close"), 2)
    latest_time = "—" if latest is None else latest.get("timestamp")
    latest_chop_state = "—" if latest is None else (latest.get("chop_state_1h") or "—")
    latest_chop_score = "—" if latest is None else fmt(latest.get("chop_score_1h"), 1)

    def df_table(df, cols=None, max_rows=50):
        if df is None or df.empty:
            return "<p class='muted'>No rows yet.</p>"
        d = df.copy().head(max_rows)
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        return d.to_html(index=False, classes="data", border=0, escape=False)

    def section(title, html, note=""):
        note_html = f"<div class='muted small'>{note}</div>" if note else ""
        return f"<details class='section'><summary>{title}</summary>{note_html}<div class='scroll'>{html}</div></details>"

    summary_html = df_table(summary, max_rows=20)
    chop_summary_html = df_table(chop_summary, max_rows=40)
    latest_html = df_table(
        latest_trades,
        cols=[
            "id", "model", "side", "signal_time", "entry_price", "chop_state", "chop_score",
            "trend_clean_flag", "failed_breakout_flag", "current_return_pct", "mfe_pct", "mae_pct",
            "ret_24h_pct", "ret_48h_pct", "ret_96h_pct", "stop_2atr_hit", "reached_1r_2atr", "status",
        ],
        max_rows=30,
    )
    events_html = df_table(latest_events, cols=["created_at", "event_type", "message"], max_rows=20)

    html = f"""
    <!doctype html>
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{APP_NAME}</title>
    <style>
      body {{ font-family: Arial, sans-serif; background:#0f1117; color:#e6e6e6; margin:0; padding:20px; }}
      h1,h2 {{ margin: 12px 0; }}
      a {{ color:#8ab4ff; }}
      .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:16px 0; }}
      .card {{ background:#171b24; border:1px solid #2a3140; border-radius:10px; padding:14px; }}
      .big {{ font-size:24px; font-weight:bold; margin-top:6px; }}
      .muted {{ color:#9aa4b2; }}
      .small {{ font-size:12px; margin:6px 0 10px; }}
      table.data {{ border-collapse:collapse; width:100%; font-size:13px; }}
      table.data th, table.data td {{ border-bottom:1px solid #2a3140; padding:7px; text-align:left; }}
      table.data th {{ color:#ffffff; background:#1e2430; position:sticky; top:0; }}
      .section {{ margin-top:14px; background:#171b24; border:1px solid #2a3140; border-radius:10px; padding:0; }}
      .section summary {{ cursor:pointer; font-weight:bold; font-size:18px; padding:14px; list-style:none; }}
      .section summary::-webkit-details-marker {{ display:none; }}
      .section summary:before {{ content:'▸'; display:inline-block; margin-right:8px; color:#8ab4ff; }}
      .section[open] summary:before {{ content:'▾'; }}
      .section .scroll {{ overflow-x:auto; padding:0 14px 14px; }}
      .pill {{ display:inline-block; padding:4px 8px; border:1px solid #2a3140; border-radius:999px; margin:4px 6px 4px 0; color:#cbd5e1; }}
      .pill-trend {{ border-color:#22c55e; }}
      .pill-mixed {{ border-color:#f59e0b; }}
      .pill-chop {{ border-color:#ef4444; }}
    </style></head><body>
    <h1>{APP_NAME}</h1>
    <div class="muted">Research only. No broker execution. Latest candle: {latest_time}</div>
    <p><a href="/health">Health</a> · <a href="/bootstrap">Bootstrap</a> · <a href="/refresh_chop">Refresh chop tags</a> · <a href="/export">Export ZIP</a></p>
    <div class="grid">
      <div class="card"><div class="muted">BTC latest close</div><div class="big">{latest_price}</div></div>
      <div class="card"><div class="muted">1h chop state</div><div class="big">{latest_chop_state}</div><div class="muted">score {latest_chop_score}</div></div>
      <div class="card"><div class="muted">Shadow trades</div><div class="big">{total_trades}</div></div>
      <div class="card"><div class="muted">Open trades</div><div class="big">{open_trades}</div></div>
      <div class="card"><div class="muted">96h completed</div><div class="big">{completed_96}</div></div>
    </div>
    <div class="card"><b>Models:</b><br>{''.join([f'<span class="pill">{m["label"]}</span>' for m in MODEL_SPECS])}</div>
    {section('Model Summary', summary_html)}
    {section('Chop / Trend-Clean Summary', chop_summary_html, 'Research tag only. Trades are still logged in TREND, MIXED, and CHOP so we can later test whether chop should block stacking.')}
    {section('Latest Shadow Trades', latest_html)}
    {section('Recent Events', events_html)}
    </body></html>
    """
    return html

init_db()
if BOOTSTRAP_ON_START:
    try:
        with connect() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM candles_1h").fetchone()["n"]
        if n == 0:
            bootstrap_yfinance(force=False)
    except Exception as e:
        log_event("bootstrap_error", str(e), {})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
