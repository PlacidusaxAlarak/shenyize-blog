# Cleveland Sample Backgrounds Design

**Date:** 2026-04-16

## Goal

Fetch a small Cleveland Museum of Art sample set of captcha backgrounds so the user can inspect source quality before deciding whether CMA should become a full production source.

## Scope

- In scope:
  - fetch exactly 5 CMA sample images
  - use only official CMA Open Access data
  - keep the images compatible with the existing captcha background pool
  - append valid sample entries to the existing background manifest
- Out of scope:
  - full multi-source crawler refactor
  - replacing the current fetch script architecture
  - batch crawling CMA to a large target count

## Constraints

- Only use official Cleveland Museum of Art Open Access endpoints.
- Only accept `CC0` records.
- Reuse the current local pool layout under `public/captcha/backgrounds/`.
- Preserve existing manifest entries.
- Keep the change small enough to validate the source, not redesign the whole crawler.

## Approach

Add a separate sample-fetch script for CMA instead of forcing the current single-source crawler into a multi-source design. The script will search a small list of place-centric keywords, filter CMA records down to CC0 scenic or architectural scenes, download 5 images into `public/captcha/backgrounds/cma/`, and append stable manifest entries.

This keeps the trial cheap: if CMA image quality is good, the repo can later promote the logic into a generalized multi-source crawler. If the source is noisy, the repo only carries a contained sample script and five sample assets.

## Filtering Strategy

The sample fetch should keep the same quality bar already used for captcha backgrounds:

- reject portrait-like records
- reject decorative-object records
- reject text-heavy or page-like presentations
- reject sparse compositions with large blank areas when the object presentation strongly suggests album or scroll style
- keep scenic, architectural, bridge, garden, courtyard, street, and place-centric records

## Data Layout

- Images: `public/captcha/backgrounds/cma/cma-<id>.jpg`
- Cache:
  - `scripts/cache/captcha/cma/search/*.json`
  - `scripts/cache/captcha/cma/artworks/*.json`
- Manifest entries should follow the existing contract, with:
  - `source: "cma"`
  - `id: "cma-<id>"`
  - `objectId: <id>`

## Verification

- Add focused tests for:
  - CMA manifest entry creation
  - CMA sample config / keyword presence
  - CMA filters retaining scenic architectural works while rejecting portrait or decorative records
- Run the focused crawler tests.
- Run the CMA sample fetcher.
- Confirm exactly 5 new CMA entries exist in the manifest and that local files exist for each.
