import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);

async function readRepoFile(relativePath) {
	const filePath = new URL(relativePath, repoRoot);
	return readFile(filePath, "utf8");
}

test("standalone slider captcha demo is removed so the site keeps only the article captcha", async () => {
	await assert.rejects(() => access(new URL("../public/slider-captcha/index.html", import.meta.url)));
	await assert.rejects(() => access(new URL("../public/slider-captcha/app.js", import.meta.url)));
});

test("projects page no longer exposes the removed slider captcha route", async () => {
	const projectsPage = await readRepoFile("src/pages/projects.astro");

	assert.doesNotMatch(projectsPage, /url\("\/slider-captcha\/"\)/);
	assert.doesNotMatch(projectsPage, /slider-captcha/);
});
