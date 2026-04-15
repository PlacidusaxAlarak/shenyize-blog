# Article Captcha Random Slider Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the article captcha start from either slider edge at random and float the entire slider control block to a random fully visible position that avoids the captcha image.

**Architecture:** Keep challenge generation and floating-position math in `src/scripts/article-captcha/logic.mjs` so the behavior stays testable with `node:test`. Update the article captcha runtime and component markup to position a floating controls wrapper from those pure helpers while preserving the existing overlay gate flow.

**Tech Stack:** Astro 5, browser DOM APIs, `node:test`

---

### Task 1: Lock behavior with failing tests

**Files:**
- Modify: `test/article-captcha-gate.test.mjs`
- Test: `test/article-captcha-gate.test.mjs`

**Step 1: Write the failing test**

Add assertions for:
- `createRotateChallenge()` starting at either `sliderMinValue` or `sliderMaxValue`
- a pure floating-controls position helper that stays inside the viewport and outside a blocked image rectangle

**Step 2: Run test to verify it fails**

Run: `node --test test/article-captcha-gate.test.mjs`
Expected: FAIL because the new helper/behavior does not exist yet

### Task 2: Implement the pure challenge/layout logic

**Files:**
- Modify: `src/scripts/article-captcha/logic.mjs`
- Modify: `src/scripts/article-captcha/logic.d.ts`

**Step 1: Write minimal implementation**

Add:
- randomized slider start edge selection
- a floating-controls placement helper with viewport padding and blocked-rect avoidance

**Step 2: Run test to verify it passes**

Run: `node --test test/article-captcha-gate.test.mjs`
Expected: PASS

### Task 3: Wire the runtime and component layout

**Files:**
- Modify: `src/scripts/article-captcha/index.ts`
- Modify: `src/components/article-captcha/ArticleCaptchaGate.astro`

**Step 1: Update markup/styles**

Create a floating controls wrapper around the refresh button, slider, and status UI so the runtime can position it anywhere in the overlay.

**Step 2: Update runtime**

Measure the card/canvas/controls, compute a safe random position, and apply it on initial load, refresh, and resize while keeping the current lock/unlock behavior.

**Step 3: Run focused verification**

Run:
- `node --test test/article-captcha-gate.test.mjs`
- `node --test`

Expected: the focused captcha tests pass; broader test results should be recorded if unrelated failures remain
