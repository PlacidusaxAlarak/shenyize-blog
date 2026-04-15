# Scenic Captcha Background Expansion Design

**Date:** 2026-04-15

## Goal

Keep the existing captcha background manifest as-is, but require that all newly added backgrounds are strongly scenic so the crawler can continue toward 1000 total entries without adding more portraits, vessels, statues, or other object-centric images.

## Constraints

- Do not rewrite or prune the existing `public/captcha/backgrounds/manifest.json` entries.
- Only `The Met Collection API` remains in scope.
- The crawler must continue to write local `.jpg` assets into `public/captcha/backgrounds/met/`.
- Existing runtime code should not change as part of this step.

## Design

### Existing manifest stays untouched

The current manifest contains a mixed pool. That is acceptable for this task because the user explicitly limited the new requirement to future additions only.

### New additions use stricter scenic admission

New candidates must pass both:

1. A stronger scenic-signal test based on object metadata, not just the search keyword.
2. A broader reject list for portraits, statues, vessels, devotional figures, furniture, and other object-centric records.

The scenic test should look at title, classification, medium, department, and tags, but should not let the keyword alone satisfy the scenic requirement.

### Search input shifts toward scenic terms

The keyword list should expand toward clearly scenic queries such as:

- `landscape`
- `landscape painting`
- `mountain landscape`
- `river landscape`
- `forest landscape`
- `seascape`
- `harbor`
- `bay`
- `canal`
- `cityscape`
- `street scene`
- `village`
- `garden`
- `park`
- `bridge`
- `ruins`
- `castle`

This increases the scenic candidate pool without changing the source policy.

### Ranking still matters after filtering

After hard filtering, the crawler should continue to rank scenic-heavy records first so that repeated continuation runs preferentially fill the remaining quota with the strongest landscape candidates.

## Verification

- Add tests that prove scenic records are accepted.
- Add tests that prove portraits and vessel-like objects are rejected even when surfaced from scenic keywords.
- Re-run the focused crawler tests.
- Re-run `pnpm build`.
- Resume the crawl and verify manifest count increases toward 1000 while keeping existing entries intact.
