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
