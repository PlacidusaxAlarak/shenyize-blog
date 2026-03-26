# Slider Captcha Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy the existing slider captcha simulator into the Astro blog at `/slider-captcha/` and add a visible entry from the Projects page.

**Architecture:** Treat the captcha as a self-contained static microsite under `public/slider-captcha/` so Astro copies it verbatim at build time. Keep the blog integration minimal by only adapting asset paths for subpath hosting and adding one curated card on the existing Projects page.

**Tech Stack:** Astro 5, static assets from `public/`, Node built-in test runner for a lightweight deployment regression test.

---

### Task 1: Add deployment regression coverage

**Files:**
- Create: `test/slider-captcha-deployment.test.mjs`

**Step 1: Write the failing test**

Create a Node test that asserts:
- `public/slider-captcha/index.html` exists and references `./styles.css` and `./app.js`
- `public/slider-captcha/js/config.js` exists and references `./assets/...`
- `src/pages/projects.astro` links to `url("/slider-captcha/")`

**Step 2: Run test to verify it fails**

Run: `node --test test/slider-captcha-deployment.test.mjs`
Expected: FAIL because the static captcha bundle is not yet present and the Projects page does not link to it.

**Step 3: Write minimal implementation**

Add the static captcha files and update `src/pages/projects.astro`.

**Step 4: Run test to verify it passes**

Run: `node --test test/slider-captcha-deployment.test.mjs`
Expected: PASS

### Task 2: Add the static captcha bundle

**Files:**
- Create: `public/slider-captcha/index.html`
- Create: `public/slider-captcha/app.js`
- Create: `public/slider-captcha/styles.css`
- Create: `public/slider-captcha/assets/demo-background.svg`
- Create: `public/slider-captcha/assets/placeholder-background.svg`
- Create: `public/slider-captcha/js/captcha-interactions.js`
- Create: `public/slider-captcha/js/captcha-logic.js`
- Create: `public/slider-captcha/js/captcha-renderer.js`
- Create: `public/slider-captcha/js/config.js`

**Step 1: Copy the existing simulator assets**

Copy the simulator's `public/` files from `G:/slider-captcha-simulator/public/` into `public/slider-captcha/`.

**Step 2: Adapt paths for subpath hosting**

Update HTML and config asset URLs from root-absolute paths to relative paths so the page works under `/slider-captcha/`.

**Step 3: Keep the page self-contained**

Avoid coupling the simulator to Astro layouts or runtime code. The page should remain directly accessible as a static route.

### Task 3: Add a Projects page entry

**Files:**
- Modify: `src/pages/projects.astro`

**Step 1: Replace placeholder content**

Add a real card for the slider captcha simulator with:
- short description
- tech tags
- a direct link to `/slider-captcha/`

**Step 2: Preserve the existing page structure**

Keep the current layout and styling conventions so this remains a normal site page, not a one-off landing page.

### Task 4: Verify the deployed route

**Files:**
- Test: `test/slider-captcha-deployment.test.mjs`

**Step 1: Run the deployment regression test**

Run: `node --test test/slider-captcha-deployment.test.mjs`
Expected: PASS

**Step 2: Run site checks**

Run: `pnpm check`
Expected: PASS

**Step 3: Run a production build**

Run: `pnpm build`
Expected: PASS and emit `dist/slider-captcha/index.html`
