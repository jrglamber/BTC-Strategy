# BTC Regime Research Logger

Research-only BTC shadow logger. No broker execution.

## Files
- `app.py` — Flask/Railway app, SQLite database, dashboard, webhook, export ZIP.
- `requirements.txt` — Python dependencies.
- `Procfile` — Railway start command.
- `btc_regime_research_logger.pine` — TradingView indicator/alert script.

## What changed in this version
- Adds BTC chop/trend-clean research tags to each 1h candle and each shadow trade.
- Does **not** block trades using chop yet. It only logs `TREND`, `MIXED`, or `CHOP` so exports can later prove whether chop should stop stacking.
- Dashboard table sections are collapsed by default.

## Railway environment variables
Set these on Railway:

- `WEBHOOK_SECRET` — any private string. Use the same value in the Pine script input.
- `DB_PATH` — `/data/btc_research.sqlite` if you add a persistent volume mounted at `/data`; otherwise leave blank for temporary storage.
- `BOOTSTRAP_ON_START` — `true`
- `BOOTSTRAP_TICKER` — `BTC-USD`
- `BOOTSTRAP_PERIOD` — `730d`

Optional chop settings, safe to leave as defaults:

- `CHOP_LOOKBACK` — default `24`
- `CHOP_MEDIAN_LOOKBACK` — default `96`
- `CHOP_TREND_MAX` — default `34`
- `CHOP_CHOP_MIN` — default `60`

## TradingView alert
1. Open a BTC 1h chart.
2. Add the Pine script.
3. Set the `Webhook secret` input to the same value as Railway `WEBHOOK_SECRET`.
4. Create alert:
   - Condition: `BTC Regime Research Logger` → `Any alert() function call`
   - Frequency: `Once Per Bar Close`
   - Webhook URL: `https://YOUR-RAILWAY-APP.up.railway.app/webhook`
5. Leave it running.

## Useful URLs
- `/` dashboard
- `/health` health/status JSON
- `/bootstrap` bootstrap historical candles if needed
- `/bootstrap?force=true` force reload Yahoo BTC candles
- `/refresh_chop` backfill/refresh chop tags on stored candles
- `/export` download ZIP of CSV exports
