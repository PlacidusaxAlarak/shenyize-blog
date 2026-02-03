# Repository Guidelines

## Project Structure & Module Organization
This is an Astro-based blog/site. Source code lives in `src/` with `components/`, `layouts/`, `pages/`, `styles/`, `utils/`, `plugins/`, `i18n/`, and `types/`. Content collections are in `src/content/` and posts live in `src/content/posts/` (schema in `src/content/config.ts`). Static assets go in `public/`. Tooling scripts live in `scripts/`, and generated output goes to `dist/`. There is also a top-level `posts/` folder used by this repo for long-form content.

## Build, Test, and Development Commands
- `pnpm install`: install deps (Node >=20, pnpm >=9; enforced by `preinstall`).
- `pnpm dev` / `pnpm start`: local dev server at `http://localhost:4321`.
- `pnpm build`: production build plus Pagefind index into `dist/`.
- `pnpm preview`: serve the built site.
- `pnpm check`: Astro checks for content/types.
- `pnpm type-check`: run `tsc` without emit.
- `pnpm lint`: Biome checks and auto-fixes `src/`.
- `pnpm format`: Biome format `src/`.
- `pnpm new-post <slug>`: scaffold a markdown post in `src/content/posts/`.

## Coding Style & Naming Conventions
Formatting is handled by Biome with tabs and double quotes. Organize imports via Biome actions. Prefer kebab-case filenames for posts (the `new-post` script uses the filename for the title) and keep frontmatter fields consistent with `frontmatter.json`.

## Testing Guidelines
No dedicated unit test framework is configured. Treat `pnpm check`, `pnpm type-check`, and `pnpm lint` as required pre-PR checks. For visual/content changes, verify locally with `pnpm dev` and a `pnpm build`/`pnpm preview` run.

## Commit & Pull Request Guidelines
Recent history uses short summary messages (often update/fix in Chinese), but `CONTRIBUTING.md` asks for Conventional Commits when possible. Prefer `feat:`, `fix:`, `docs:`, `refactor:`, etc. Keep PRs focused on a single purpose, link related issues/discussions for major changes, and include screenshots or GIFs for UI/layout edits.

## Configuration & Content Tips
Primary site settings live in `src/config.ts`; deployment URL and integrations are in `astro.config.mjs`. If you modify frontmatter fields, update `frontmatter.json` and `src/content/config.ts` together. Adjust search indexing via `pagefind.yml` when changing build output.
