# Scenic Captcha Background Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the existing captcha background manifest intact while ensuring all newly added entries are strongly scenic and continue the crawl toward 1000 total entries.

**Architecture:** Tighten crawler admission rules so new candidates must satisfy scenic metadata checks that do not rely on the search keyword alone. Expand the Met search keyword list toward scenic phrases, then continue incremental crawl runs until the manifest reaches the configured target or a hard blocker appears.

**Tech Stack:** Node.js 20+, native `node:test`, Astro static build, Met Collection API, local JSON manifest, PowerShell shell

---

### Task 1: Add failing scenic-admission tests

**Files:**
- Modify: `test/captcha-background-fetch.test.mjs`
- Test: `test/captcha-background-fetch.test.mjs`

**Step 1: Write a failing test for scenic object acceptance**

Add a focused test that imports the crawler helper and asserts a clearly scenic record passes the stricter admission rules.

**Step 2: Run test to verify it fails**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: FAIL because the scenic helper is not exported or the current rules are still too loose.

**Step 3: Write a failing test for portrait or vessel rejection**

Add a test showing that a portrait-like or vessel-like record is rejected even if the crawler keyword is scenic.

**Step 4: Run test to verify it fails for the expected reason**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: FAIL because the current helper still accepts weak or object-centric results.

### Task 2: Tighten crawler scenic filtering

**Files:**
- Modify: `scripts/fetch-captcha-backgrounds.mjs`
- Test: `test/captcha-background-fetch.test.mjs`

**Step 1: Export a scenic-admission helper**

Implement and export a helper that evaluates object metadata for:

- scenic-positive title and tag patterns
- scenic-negative portrait and object patterns
- object-centric classification rejections

**Step 2: Ensure keyword text alone cannot satisfy the scenic requirement**

Use metadata-only text for the scenic threshold so a scenic search term does not automatically admit unrelated objects.

**Step 3: Reuse the same logic in crawl-time filtering and priority scoring**

Make the main crawler path call the stricter helper before download.

**Step 4: Run the focused tests**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS

### Task 3: Expand scenic search keywords

**Files:**
- Modify: `scripts/fetch-captcha-backgrounds.config.json`
- Test: `test/captcha-background-fetch.test.mjs`

**Step 1: Update the config keyword list with more scenic phrases**

Keep `targetCount` at `1000`, but broaden the keyword list toward scenic terms such as `seascape`, `harbor`, `bay`, `cityscape`, `street scene`, `park`, and `ruins`.

**Step 2: Extend the config test**

Assert the config still targets the same output directories and now includes the added scenic phrases.

**Step 3: Run the focused tests**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS

### Task 4: Verify code before crawl continuation

**Files:**
- Test: `test/captcha-background-fetch.test.mjs`
- Test: `scripts/fetch-captcha-backgrounds.mjs`
- Test: `scripts/fetch-captcha-backgrounds.config.json`

**Step 1: Run the focused crawler test suite**

Run: `node --test test/captcha-background-fetch.test.mjs`

Expected: PASS

**Step 2: Run the production build**

Run: `pnpm build`

Expected: PASS

### Task 5: Continue the crawl toward 1000 total entries

**Files:**
- Modify: `public/captcha/backgrounds/manifest.json`
- Modify: `public/captcha/backgrounds/met/*.jpg`
- Modify: `scripts/cache/captcha/met/search/*.json`
- Modify: `scripts/cache/captcha/met/objects/*.json`

**Step 1: Run the crawler**

Run: `node scripts/fetch-captcha-backgrounds.mjs`

Expected: new scenic entries are added while existing manifest entries remain untouched.

**Step 2: Measure progress**

Run: `node -e "const fs=require('fs'); const manifest=JSON.parse(fs.readFileSync('public/captcha/backgrounds/manifest.json','utf8')); console.log(manifest.length)"`

Expected: manifest count increases.

**Step 3: Repeat continuation runs as needed**

Re-run the crawler until the manifest reaches `1000` or a hard blocker appears.

**Step 4: Re-run build verification after the crawl**

Run: `pnpm build`

Expected: PASS and static assets appear under `dist/captcha/backgrounds/`.

### Task 6: Record the continuation result

**Files:**
- Modify: `docs/plans/2026-04-15-captcha-background-crawl.md`

**Step 1: Append scenic-expansion execution notes**

Record:

- starting and ending manifest counts
- updated scenic keyword set
- whether the 1000 target was reached
- any stable Met download failures
- any remaining blockers

**Step 2: Commit if requested**

```bash
git add scripts/fetch-captcha-backgrounds.mjs scripts/fetch-captcha-backgrounds.config.json test/captcha-background-fetch.test.mjs public/captcha/backgrounds docs/plans/2026-04-15-captcha-background-crawl.md docs/plans/2026-04-15-scenic-captcha-background-expansion-design.md docs/plans/2026-04-15-scenic-captcha-background-expansion.md
git commit -m "feat: prefer scenic captcha backgrounds"
```
