# Cleveland Sample Backgrounds Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal Cleveland Museum of Art sample fetch flow that downloads 5 CC0 captcha background images into the existing pool.

**Architecture:** Keep the current single-source background crawler intact and add a separate CMA sample script for this experiment. Reuse the existing manifest contract and the current scenic / sparse-composition filters so the downloaded images can participate in the captcha pool immediately without a larger multi-source refactor.

**Tech Stack:** Node.js 20+, native `fetch`, `sharp`, JSON manifest, `node:test`

---

### Task 1: Add focused CMA sample tests

**Files:**
- Modify: `test/captcha-background-fetch.test.mjs`
- Test: `test/captcha-background-fetch.test.mjs`

**Step 1: Write the failing test**

Add tests that assert:

- a CMA sample config or keyword list exists in the new script
- CMA manifest entries use:
  - `id: "cma-<id>"`
  - `source: "cma"`
  - `localPath: "/captcha/backgrounds/cma/cma-<id>.jpg"`
- CMA filtering accepts a clearly scenic architectural CMA-like record

**Step 2: Run test to verify it fails**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: FAIL because CMA support does not exist yet.

### Task 2: Implement the minimal CMA sample fetcher

**Files:**
- Create: `scripts/fetch-cma-sample-backgrounds.mjs`
- Modify: `scripts/fetch-captcha-backgrounds.mjs`
- Test: `test/captcha-background-fetch.test.mjs`

**Step 1: Implement CMA API helpers**

Add minimal helpers for:

- CMA search requests
- CMA artwork detail retrieval
- CMA image URL selection
- CMA manifest entry creation

**Step 2: Reuse the current quality filters**

Use the existing scenic and sparse-composition helpers where possible so sample quality stays aligned with the current captcha dataset.

**Step 3: Limit output to 5**

The CMA script should:

- search a small set of keywords such as `landscape`, `architecture`, `bridge`, `street`, `garden`
- fetch and filter candidate records
- stop after 5 accepted images

**Step 4: Run focused test**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS.

### Task 3: Execute the sample fetch

**Files:**
- Modify: `public/captcha/backgrounds/manifest.json`
- Create: `public/captcha/backgrounds/cma/*.jpg`
- Create: `scripts/cache/captcha/cma/search/*.json`
- Create: `scripts/cache/captcha/cma/artworks/*.json`

**Step 1: Run the sample fetcher**

Run: `node scripts/fetch-cma-sample-backgrounds.mjs`

Expected:

- 5 CMA images are downloaded
- 5 `cma-*` entries are appended to the manifest
- local images exist under `public/captcha/backgrounds/cma/`

**Step 2: Verify manifest consistency**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS.

### Task 4: Record the sample outcome

**Files:**
- Modify: `docs/plans/2026-04-16-cma-sample-backgrounds-design.md`

**Step 1: Append execution notes**

Record:

- the 5 downloaded object IDs
- any CMA-specific filtering surprises
- whether the source quality looks promising enough for full integration

**Step 2: Commit**

```bash
git add docs/plans/2026-04-16-cma-sample-backgrounds-design.md docs/plans/2026-04-16-cma-sample-backgrounds.md scripts/fetch-cma-sample-backgrounds.mjs scripts/fetch-captcha-backgrounds.mjs test/captcha-background-fetch.test.mjs public/captcha/backgrounds/manifest.json public/captcha/backgrounds/cma scripts/cache/captcha/cma
git commit -m "feat: add cma captcha background sample fetch"
```
