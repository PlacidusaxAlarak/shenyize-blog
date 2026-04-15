# Captcha Background Crawl Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local captcha background image pool for `shenyize-blog` by downloading allowed images from official open-access sources, storing them under `public/captcha/backgrounds/`, and generating a manifest that later runtime code can randomly select from.

**Architecture:** Keep phase 1 intentionally narrow: implement one fetch pipeline against `The Met Collection API`, save normalized files into the blog's public asset tree, and emit a JSON manifest with enough metadata for later filtering and runtime selection. Do not change captcha geometry or page-gating behavior in this plan; this plan only prepares a stable local asset pool and the crawler needed to refresh it.

**Tech Stack:** Node.js 20+, Astro 5 repo layout, PowerShell shell, JSON manifest, native `fetch`, native `node:test`

---

## Scope

- In scope:
  - official-API image fetching
  - local asset directory creation
  - manifest generation
  - basic regression coverage for manifest shape and file output
- Out of scope:
  - scraping arbitrary HTML pages
  - runtime captcha image randomization
  - cleanup of existing repo-wide type or Astro check failures
  - deleting or replacing `public/slider-captcha/`

## Current Repo Context

- Existing captcha assets live under `public/captcha/`
- Current single-image default is wired through:
  - `src/components/article-captcha/ArticleCaptchaGate.astro`
  - `src/scripts/article-captcha/index.ts`
- Existing placeholder fallback is `public/captcha/placeholder-background.svg`
- Existing independent demo under `public/slider-captcha/` must remain untouched

## Data Source Policy

### Phase 1 Source

- Use only `The Met Collection API`
- API docs: `https://metmuseum.github.io/`
- Open-access dataset reference: `https://github.com/metmuseum/openaccess`

### Phase 2 Candidates

- `Smithsonian Open Access`, but only after an API key is available
- `Rijksmuseum Data Services`, only as a later expansion task

### Forbidden Sources For This Plan

- Unsplash
- Pexels
- Pixabay
- Openverse catalog pages
- Google Images / Bing Images / Pinterest / social sites
- any normal HTML page scraping

## Image Selection Rules

### Preferred Subjects

- landscape
- architecture
- city view
- street scene
- interior
- garden
- still life
- vase
- flower
- temple
- bridge
- painting
- engraving

### Reject Heuristics

- center area is mostly blank
- image is strongly symmetrical
- image is mostly sky, sea, wall, paper, or other low-detail flat surface
- large text dominates the center
- visible watermark or logo
- no usable primary image URL in source metadata

### Minimum Technical Quality

- at least one image dimension `>= 1000px`
- save local copies as `.jpg`
- skip duplicate `objectId`
- record enough metadata to later filter or re-rank images without re-fetching

## Target Directory Layout

```text
public/
  captcha/
    backgrounds/
      manifest.json
      met/
        met-<objectId>.jpg

scripts/
  fetch-captcha-backgrounds.mjs
  fetch-captcha-backgrounds.config.json
  cache/
    captcha/
```

## Manifest Contract

Write `public/captcha/backgrounds/manifest.json` as a flat JSON array. Every entry must contain:

```json
{
  "id": "met-436121",
  "source": "met",
  "objectId": 436121,
  "title": "Sunflowers",
  "imageUrl": "https://...",
  "localPath": "/captcha/backgrounds/met/met-436121.jpg",
  "objectUrl": "https://...",
  "license": "CC0",
  "width": 0,
  "height": 0,
  "tags": ["painting", "flower"],
  "fetchedAt": "2026-04-15T00:00:00.000Z"
}
```

Notes:

- `localPath` must be web-ready, not a filesystem path
- `license` should be normalized to a stable string such as `CC0`
- `tags` should contain the keyword that surfaced the image plus obvious subject labels when available
- entries should be sorted deterministically, preferably by `source` then `objectId`

## Fetch Config Contract

Create `scripts/fetch-captcha-backgrounds.config.json` with:

```json
{
  "sources": ["met"],
  "keywords": [
    "landscape",
    "architecture",
    "city",
    "street",
    "interior",
    "garden",
    "still life",
    "vase",
    "flower",
    "temple",
    "bridge"
  ],
  "targetCount": 120,
  "maxPerKeyword": 40,
  "minLongEdge": 1000,
  "outputDir": "public/captcha/backgrounds",
  "cacheDir": "scripts/cache/captcha"
}
```

## Script Behavior Contract

The fetcher at `scripts/fetch-captcha-backgrounds.mjs` must:

1. Read the config file.
2. Create missing directories under `public/captcha/backgrounds/` and `scripts/cache/captcha/`.
3. Query The Met search endpoint for each keyword.
4. Merge and deduplicate returned `objectId` values.
5. Request object details one by one or in controlled batches.
6. Filter out non-open, missing-image, low-detail, and low-resolution records.
7. Download the primary image into `public/captcha/backgrounds/met/`.
8. Generate `manifest.json`.
9. Print a summary report with:
   - keywords processed
   - candidate ids found
   - detail records fetched
   - images downloaded
   - duplicates skipped
   - filtered counts by reason

### Implementation Constraints

- Use official JSON APIs only
- Prefer native Node APIs over extra dependencies
- Be tolerant of individual download failures
- Continue on partial failures and report them at the end
- Do not overwrite unrelated files
- Do not modify existing captcha runtime files in this plan

## Task 1: Add crawl regression coverage

**Files:**
- Create: `test/captcha-background-fetch.test.mjs`
- Test: `test/captcha-background-fetch.test.mjs`

**Step 1: Write the failing test**

Add tests that assert:

- the fetch config file exists and contains at least the required keywords and `targetCount`
- the planned output directory path is `public/captcha/backgrounds`
- a generated manifest entry shape includes:
  - `id`
  - `source`
  - `objectId`
  - `localPath`
  - `license`
  - `fetchedAt`

**Step 2: Run test to verify it fails**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: FAIL because the config and manifest contract implementation do not exist yet

## Task 2: Create config and fetch script

**Files:**
- Create: `scripts/fetch-captcha-backgrounds.config.json`
- Create: `scripts/fetch-captcha-backgrounds.mjs`

**Step 1: Write minimal configuration**

Create the JSON config with:

- `sources: ["met"]`
- the approved keyword list
- `targetCount: 120`
- `maxPerKeyword: 40`
- `minLongEdge: 1000`

**Step 2: Write minimal fetcher implementation**

Implement:

- config loading
- directory creation
- Met search requests
- Met object detail requests
- filtering
- file download
- manifest write
- terminal summary logging

**Step 3: Run focused test**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS

## Task 3: Execute the first crawl

**Files:**
- Create: `public/captcha/backgrounds/manifest.json`
- Create: `public/captcha/backgrounds/met/*.jpg`

**Step 1: Run the fetcher**

Run: `node scripts/fetch-captcha-backgrounds.mjs`

Expected:

- directories are created if missing
- candidate objects are fetched from The Met
- around `80-150` valid images are downloaded
- `manifest.json` is written

**Step 2: Validate output count**

Run: `Get-ChildItem -Path 'public/captcha/backgrounds/met' -File | Measure-Object`

Expected: Count is at least `80`

**Step 3: Spot-check manifest**

Run: `Get-Content -Path 'public/captcha/backgrounds/manifest.json' -TotalCount 40`

Expected: JSON array with stable fields and web-ready `localPath` values such as `/captcha/backgrounds/met/met-123456.jpg`

## Task 4: Verify asset integrity

**Files:**
- Test: `public/captcha/backgrounds/manifest.json`
- Test: `public/captcha/backgrounds/met/*.jpg`

**Step 1: Run targeted tests**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS

**Step 2: Verify build still sees static assets**

Run: `pnpm build`

Expected:

- Astro build succeeds
- no changes are required in captcha runtime yet
- downloaded images are copied into `dist/captcha/backgrounds/`

## Task 5: Record crawl output for handoff

**Files:**
- Modify: `docs/plans/2026-04-15-captcha-background-crawl.md`

**Step 1: Append execution notes**

After the first successful run, append:

- actual downloaded count
- rejected count summary
- any keywords that performed poorly
- whether follow-up source expansion is necessary

**Step 2: Commit**

```bash
git add scripts/fetch-captcha-backgrounds.mjs scripts/fetch-captcha-backgrounds.config.json test/captcha-background-fetch.test.mjs public/captcha/backgrounds/manifest.json public/captcha/backgrounds/met docs/plans/2026-04-15-captcha-background-crawl.md
git commit -m "feat: add captcha background fetch pipeline"
```

## Notes For The Next Task

- The next implementation task, after this plan is complete, is runtime random selection from the local manifest or an equivalent build-time image list.
- Fallback behavior must remain `public/captcha/placeholder-background.svg` when the pool is empty or an image load fails.
- Avoid changing `public/slider-captcha/`.

Plan complete and saved to `docs/plans/2026-04-15-captcha-background-crawl.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?

## Execution Notes

### 2026-04-15 Continuation Run

- Continued the crawl in the existing workspace state instead of a fresh worktree because the repo already contained in-progress crawler output, cache files, and uncommitted assets that needed to be resumed in place.
- Starting state before this continuation:
  - `public/captcha/backgrounds/manifest.json`: `184` entries
  - `public/captcha/backgrounds/met/`: `204` `.jpg` files
- After three additional successful continuation runs:
  - `public/captcha/backgrounds/manifest.json`: `413` entries
  - `public/captcha/backgrounds/met/`: `433` `.jpg` files
  - `dist/captcha/backgrounds/met/`: `433` `.jpg` files after build
  - `dist/captcha/backgrounds/manifest.json`: present
- Fresh verification completed:
  - `node --test test/captcha-background-fetch.test.mjs`: PASS (`4/4`)
  - `pnpm build`: PASS
  - manifest parse check: PASS
  - manifest-to-file integrity check: `0` missing files for manifest entries
- Current residual inconsistency:
  - `20` local `.jpg` files exist under `public/captcha/backgrounds/met/` but are not referenced by the current manifest
  - These appear to be leftovers from earlier partial runs; the current manifest itself is internally consistent
- Most recent crawl summary:
  - `Keywords processed: 13`
  - `Candidate IDs found: 3492`
  - `Detail records fetched: 534`
  - `Images downloaded: 59`
  - `Duplicates skipped: 2295`
  - `Filtered detail_fetch_failed: 663`
  - `Filtered image_download_failed: 6`
  - `Filtered low_relevance: 42`
  - `Filtered low_resolution: 6`
  - `Filtered not_public_domain: 67`
- Known hard download failures are currently persistent `404` image URLs for Met object IDs:
  - `319873`
  - `464294`
  - `549493`
  - `549494`
  - `623394`
  - `623395`
- Keywords that performed relatively weakly in Met search result volume during this session:
  - `waterfall` (`136`)
  - `countryside` (`146`)
  - `forest` (`180`)
- Source-expansion assessment:
  - not necessary for the next runtime-randomization task
  - worth revisiting only after stabilizing Met detail-fetch success rate and deciding whether the current `413` manifest-backed images are sufficient
