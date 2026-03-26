import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);

async function readRepoFile(relativePath) {
	const filePath = new URL(relativePath, repoRoot);
	return readFile(filePath, "utf8");
}

test("slider captcha static bundle is hosted from a subpath with relative asset URLs", async () => {
	const indexHtml = await readRepoFile("public/slider-captcha/index.html");
	const configJs = await readRepoFile("public/slider-captcha/js/config.js");

	assert.match(indexHtml, /href="\.\/styles\.css"/);
	assert.match(indexHtml, /src="\.\/app\.js"/);
	assert.doesNotMatch(indexHtml, /href="\/styles\.css"/);
	assert.doesNotMatch(indexHtml, /src="\/app\.js"/);

	assert.match(configJs, /backgroundImageUrl:\s*"\.\/assets\/demo-background\.svg"/);
	assert.match(configJs, /fallbackBackgroundImageUrl:\s*"\.\/assets\/placeholder-background\.svg"/);
});

test("projects page exposes a link to the deployed slider captcha route", async () => {
	const projectsPage = await readRepoFile("src/pages/projects.astro");

	assert.match(projectsPage, /url\("\/slider-captcha\/"\)/);
	assert.match(projectsPage, /滑动拼图验证码仿真/);
});
