# v1 transport patch — let Railway fetch today.json over HTTPS

This is the one-time edit to v1 (`Market Temp Gauge` repo) that lets it
read v2's daily snapshot when v1 is running on Railway and v2 is
running on GitHub Actions.

The existing file-based reader still works for local dev — the URL path
is only used when `MTG_V2_TODAY_URL` is set in the Railway environment.

## What changes

One file: `src/sources/mtg-v2-analog.ts` (v1 repo).

Add a URL-fetch branch to `loadMtgV2Snapshot()`. Everything downstream
(staleness check, schema validation, prompt formatting) is unchanged.

## Edit

Replace `loadMtgV2Snapshot()` with the version below. The rest of the
file (interfaces, `formatSnapshotForPrompt`, etc.) stays exactly as is.

```ts
/**
 * Load v2's daily snapshot. Returns the parsed object and a status reason.
 * Never throws — failures fall back to { snapshot: null, reason: ... }.
 *
 * Source resolution (in order):
 *   1. MTG_V2_TODAY_URL — fetch over HTTPS. Used on Railway.
 *   2. MTG_V2_TODAY_PATH — read from local filesystem. Override.
 *   3. ../mtg-v2/data/today.json — default, used in local dev.
 */
export async function loadMtgV2Snapshot(): Promise<LoadResult> {
  const url = process.env.MTG_V2_TODAY_URL;

  if (url) {
    return loadFromUrl(url);
  }

  const snapshotPath =
    process.env.MTG_V2_TODAY_PATH || defaultSnapshotPath();
  return loadFromFile(snapshotPath);
}

async function loadFromUrl(url: string): Promise<LoadResult> {
  let res: Response;
  try {
    // Node 20 has global fetch.
    res = await fetch(url, {
      // Don't get a cached 304 from a CDN edge; we want a fresh read.
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" },
    });
  } catch {
    return { snapshot: null, reason: "missing", path: url, ageHours: null };
  }

  if (!res.ok) {
    return { snapshot: null, reason: "missing", path: url, ageHours: null };
  }

  let parsed: any;
  let raw: string;
  try {
    raw = await res.text();
    parsed = JSON.parse(raw);
  } catch {
    return { snapshot: null, reason: "parse_error", path: url, ageHours: null };
  }

  // Staleness check uses as_of_date from the JSON itself, since fetch
  // doesn't give us a meaningful Last-Modified on raw.githubusercontent.com
  // (it'd reflect the commit, which is close enough but flaky).
  const asOf = parsed?.as_of_date;
  if (typeof asOf !== "string") {
    return { snapshot: null, reason: "schema_error", path: url, ageHours: null };
  }
  const asOfMs = Date.parse(`${asOf}T21:30:00Z`); // publish time UTC
  const ageHours = (Date.now() - asOfMs) / 3600 / 1000;
  if (ageHours * 3600 * 1000 > maxAgeMs()) {
    return { snapshot: null, reason: "stale", path: url, ageHours };
  }

  if (!validateSchema(parsed)) {
    return { snapshot: null, reason: "schema_error", path: url, ageHours };
  }

  return { snapshot: parsed as AnalogSnapshot, reason: "ok", path: url, ageHours };
}

function loadFromFile(snapshotPath: string): LoadResult {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(snapshotPath);
  } catch {
    return { snapshot: null, reason: "missing", path: snapshotPath, ageHours: null };
  }

  const ageMs = Date.now() - stat.mtimeMs;
  const ageHours = ageMs / 3600 / 1000;
  if (ageMs > maxAgeMs()) {
    return { snapshot: null, reason: "stale", path: snapshotPath, ageHours };
  }

  let raw: string;
  try {
    raw = fs.readFileSync(snapshotPath, "utf8");
  } catch {
    return { snapshot: null, reason: "parse_error", path: snapshotPath, ageHours };
  }

  let parsed: any;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { snapshot: null, reason: "parse_error", path: snapshotPath, ageHours };
  }

  if (!validateSchema(parsed)) {
    return { snapshot: null, reason: "schema_error", path: snapshotPath, ageHours };
  }

  return { snapshot: parsed as AnalogSnapshot, reason: "ok", path: snapshotPath, ageHours };
}

function validateSchema(parsed: any): boolean {
  return !!(
    parsed &&
    typeof parsed.as_of_date === "string" &&
    parsed.regime?.label &&
    parsed.closest_analog?.date &&
    parsed.base_rates?.returns
  );
}
```

## Callers update

Anywhere that does `const { snapshot } = loadMtgV2Snapshot();` now needs
`await`: `const { snapshot } = await loadMtgV2Snapshot();`. Single call
site in `src/server.ts` — confirm with `grep -rn loadMtgV2Snapshot src/`.

## Env vars on Railway

Set on the v1 Railway service:

```
MTG_V2_TODAY_URL=https://raw.githubusercontent.com/<your-gh-user>/mtg-v2/main/data/today.json
MTG_V2_MAX_AGE_HOURS=48   # cron is weekdays; Mon morning is ~60h after Fri close
```

If you flip to Supabase Storage later, swap the URL for the bucket's
public object URL. No other changes needed.

## Local dev still works

If `MTG_V2_TODAY_URL` is unset, the resolver falls through to
`MTG_V2_TODAY_PATH` and then the sibling-directory default. So `npm run
dev` keeps working exactly as it does today.
