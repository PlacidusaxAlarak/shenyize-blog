import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);

async function readRepoFile(relativePath) {
	const filePath = new URL(relativePath, repoRoot);
	return readFile(filePath, "utf8");
}

test("posts detail page wraps article content with the article captcha gate component", async () => {
	const postsPage = await readRepoFile("src/pages/posts/[...slug].astro");

	assert.match(postsPage, /import ArticleCaptchaGate from "@components\/article-captcha\/ArticleCaptchaGate\.astro";/);
	assert.match(postsPage, /<ArticleCaptchaGate>/);
	assert.match(postsPage, /<\/ArticleCaptchaGate>/);
});

test("article captcha gate component exposes data attributes and default captcha asset paths", async () => {
	const gateComponent = await readRepoFile("src/components/article-captcha/ArticleCaptchaGate.astro");

	assert.match(gateComponent, /storageKey = "article-captcha:posts"/);
	assert.match(gateComponent, /backgroundImageUrl = "\/captcha\/demo-background\.svg"/);
	assert.match(gateComponent, /fallbackBackgroundImageUrl = "\/captcha\/placeholder-background\.svg"/);
	assert.match(gateComponent, /data-article-captcha-gate/);
	assert.match(gateComponent, /data-article-captcha-overlay/);
	assert.match(gateComponent, /\.article-captcha-overlay\[hidden\]\s*\{\s*display:\s*none;/);
});

test("article captcha gate keeps the verifier embedded above blurred article content until solved", async () => {
	const gateComponent = await readRepoFile("src/components/article-captcha/ArticleCaptchaGate.astro");

	assert.ok(
		gateComponent.indexOf("data-article-captcha-overlay") <
			gateComponent.indexOf("data-article-captcha-content"),
	);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*position:\s*sticky;/);
	assert.match(
		gateComponent,
		/\[data-gate-state="locked"\]\s*\[data-article-captcha-content\]\s*\{[\s\S]*filter:\s*blur\(/,
	);
	assert.match(
		gateComponent,
		/\[data-gate-state="locked"\]\s*\[data-article-captcha-content\]::after\s*\{/,
	);
});

test("article captcha logic keeps pentagon geometry and tolerance-based validation available to the gate", async () => {
	const logicModule = await import("../src/scripts/article-captcha/logic.mjs");
	const geometry = logicModule.createChallengeGeometry({
		canvasWidth: 360,
		canvasHeight: 220,
		pieceRadius: 34,
		sliderStartX: 24,
		padding: 18,
		rng: () => 0.5,
	});

	assert.equal(geometry.shape.points.length, 5);
	assert.equal(geometry.targetNotch.rotation, geometry.pieceRotation);
	assert.notEqual(geometry.decoyNotch.rotation, geometry.pieceRotation);
	assert.ok(geometry.maxTravel >= geometry.targetX - geometry.sliderStartX);

	assert.deepEqual(
		logicModule.evaluateAttempt({ pieceX: 140, targetX: 145, tolerancePx: 5 }),
		{ success: true, delta: 5 },
	);
	assert.deepEqual(
		logicModule.createFreshCaptchaState({ sliderStartX: 24 }),
		{
			currentPieceX: 24,
			sliderValue: 0,
			isAnimating: false,
			isLocked: false,
			status: "idle",
		},
	);
});

test("article captcha background assets are published from the blog public directory", async () => {
	await access(new URL("../public/captcha/demo-background.svg", import.meta.url));
	await access(new URL("../public/captcha/placeholder-background.svg", import.meta.url));
});

test("article captcha runtime waits 500ms after success before unlocking the gate", async () => {
	const runtime = await readRepoFile("src/scripts/article-captcha/index.ts");

	assert.match(runtime, /const SUCCESS_DISMISS_DELAY_MS = 500;/);
	assert.match(runtime, /await pause\(SUCCESS_DISMISS_DELAY_MS\);/);
});

test("article captcha runtime keeps the verifier embedded inside the article gate", async () => {
	const runtime = await readRepoFile("src/scripts/article-captcha/index.ts");

	assert.doesNotMatch(runtime, /function mountOverlayInBody\(/);
	assert.doesNotMatch(runtime, /document\.body\.appendChild\(elements\.overlay\);/);
});
