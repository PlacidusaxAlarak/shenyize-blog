# Errors

## [ERR-20260312-001] apply_patch

**Logged**: 2026-03-12T08:41:18.918942+00:00
**Priority**: medium
**Status**: pending
**Area**: docs

### Summary
`apply_patch` failed with exit code 1 and no error output in this workspace, so markdown edits needed a script fallback.

### Error
```
Exit code: 1
Output: <empty>
```

### Context
- Command/operation attempted: `functions.apply_patch`
- Input or parameters used: update to `src/content/posts/Atcoder/ABC448_D.md` and a temporary probe file
- Environment details: Codex desktop on Windows, workspace `G:\shenyize-blog`

### Suggested Fix
Investigate whether the local `apply_patch` integration is broken in this session or on this platform; fall back to scripted edits if it recurs.

### Metadata
- Reproducible: unknown
- Related Files: src/content/posts/Atcoder/ABC448_D.md

---

## [ERR-20260418-001] openimages-sample-fetch

**Logged**: 2026-04-18T09:14:20.1483193+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Open Images sample expansion failed mid-run because a Flickr-backed source URL returned `410 Gone`, which means fixed target lists are not enough on their own.

### Error
```
Error: Unable to fetch https://c7.staticflickr.com/4/3427/3272414398_ef2c8029c7_z.jpg: 410 Gone
```

### Context
- Command/operation attempted: `node scripts/fetch-openimages-sample-backgrounds.mjs`
- Input or parameters used: expand `public/openimages-sample/` from 20 to 200 images
- Environment details: Windows PowerShell, workspace `G:\shenyize-blog`

### Suggested Fix
Make the fetcher skip failed image URLs and keep selecting replacement candidates until the requested total is reached, instead of aborting the whole crawl on the first dead Flickr asset.

### Metadata
- Reproducible: yes
- Related Files: scripts/fetch-openimages-sample-backgrounds.mjs

---
