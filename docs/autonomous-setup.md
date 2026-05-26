# Autonomous publish — one-time setup

Get v2 running daily on GitHub's servers, with no local PC and no Claude
needed. ~15 minutes start to finish.

## Step 1 — Create the GitHub repo (browser)

1. Go to https://github.com/new
2. Owner: your account (`Deets2412`).
3. Name: `mtg-v2`.
4. **Visibility:**
   - **Public** → simplest. v1 can fetch `today.json` from
     `raw.githubusercontent.com` with zero auth. Recommended unless
     you have a reason otherwise — the corpus is just public market
     data, the code isn't a moat.
   - **Private** → fine too, but you'll need Supabase Storage (or
     similar) as the transport. See `v1-transport-patch.md`.
5. Do NOT initialise with README/license/.gitignore — we already have them.
6. Click **Create repository**, copy the SSH or HTTPS URL it shows.

## Step 2 — Initialise local git and push (terminal)

From `C:\Users\User\Projects\mtg-v2`:

```bash
git init
git branch -M main
git add .
git commit -m "initial: MTG v2 retrieval engine"
git remote add origin https://github.com/Deets2412/mtg-v2.git
git push -u origin main
```

The `.gitignore` already excludes the big corpus parquet and the venv.
What gets pushed: source code, calibration/regime models, today.json,
docs.

## Step 3 — Add secrets (browser)

Repo → **Settings → Secrets and variables → Actions → New repository
secret**. Add these (skip any you don't use):

| Secret name | Value | Required? |
|---|---|---|
| `RESEND_API_KEY` | from your Resend account | only if you want the daily email |
| `REPORT_EMAIL_FROM` | e.g. `mtg@yourdomain.com` | with Resend |
| `REPORT_EMAIL_TO` | your inbox | with Resend |
| `REPORT_EMAIL_BCC` | optional | with Resend |
| `SUPABASE_URL` | your Supabase project URL | only if using Supabase transport |
| `SUPABASE_SERVICE_ROLE_KEY` | from Supabase API settings | only if using Supabase transport |
| `SUPABASE_BUCKET` | bucket name, default `mtg-v2` | only if using Supabase transport |

The workflow checks each one — missing secrets just skip that branch
silently.

## Step 4 — Trigger a manual run (browser)

Repo → **Actions tab → publish-today workflow → Run workflow → main**.

Watch the run. ~3-4 minutes for the first run (cold pip cache), ~1-2
minutes thereafter. When it goes green, look at the latest commit on
`main` — there should be a `publish: today.json YYYY-MM-DD` commit from
`mtg-v2-bot`.

If it goes red, click into the failing step. yfinance occasionally
fails transiently — re-running usually fixes it.

## Step 5 — Wire v1 to fetch from the URL

See `v1-transport-patch.md` for the code edit. On Railway, set:

```
MTG_V2_TODAY_URL=https://raw.githubusercontent.com/Deets2412/mtg-v2/main/data/today.json
MTG_V2_MAX_AGE_HOURS=48
```

After Railway redeploys, the next `/api/scan` call will pull today.json
over HTTPS instead of looking at the (now non-existent) sibling
filesystem path. The reader silently falls back to "no snapshot" if the
URL is unreachable, so worst case v1 behaves as it did before v2 was
wired in.

## Step 6 — Forget about it

The workflow runs at 21:30 UTC, Mon-Fri. Your PC stays off. Claude
stays unloaded. The Actions tab shows a green check for every run; if
one goes red, GitHub emails you.

## Verification

- After step 4, the URL `https://raw.githubusercontent.com/Deets2412/mtg-v2/main/data/today.json`
  should return valid JSON in any browser tab.
- `as_of_date` in that JSON should be today (or last trading day).
- After step 5 + Railway redeploy, the next v1 scan should show the v2
  analog block (look for `=== EMPIRICAL HISTORICAL ANALOG ===` in the
  scan's prompt audit).

## Cost

- GitHub Actions on a public repo: **free, unlimited**.
- GitHub Actions on a private repo: 2,000 free min/month. This
  workflow uses ~2 min/run × ~22 runs/month = **~44 min/month**.
  Effectively free.
- Supabase Storage (if used): free tier covers this volume by ~1000x.
