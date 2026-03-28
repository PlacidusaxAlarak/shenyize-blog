import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);

async function readRepoFile(relativePath) {
	const filePath = new URL(relativePath, repoRoot);
	return readFile(filePath, "utf8");
}

test("layout mounts the sitewide captcha gate around the main slot", async () => {
	const layout = await readRepoFile("src/layouts/Layout.astro");

	assert.match(
		layout,
		/import ArticleCaptchaGate from "@components\/article-captcha\/ArticleCaptchaGate\.astro";/,
	);
	assert.match(layout, /<ArticleCaptchaGate>/);
	assert.match(layout, /<slot \/>/);
	assert.match(layout, /<\/ArticleCaptchaGate>/);
	assert.ok(layout.indexOf("<ArticleCaptchaGate>") < layout.indexOf("<slot />"));
});

test("shared pages inherit the captcha gate from layouts instead of page-local wrappers", async () => {
	const homePage = await readRepoFile("src/pages/[...page].astro");
	const postsPage = await readRepoFile("src/pages/posts/[...slug].astro");

	assert.match(homePage, /import MainGridLayout from "..\/layouts\/MainGridLayout\.astro";/);
	assert.match(homePage, /<MainGridLayout>/);
	assert.doesNotMatch(
		homePage,
		/import ArticleCaptchaGate from "@components\/article-captcha\/ArticleCaptchaGate\.astro";/,
	);
	assert.doesNotMatch(homePage, /<ArticleCaptchaGate>/);
	assert.doesNotMatch(
		postsPage,
		/import ArticleCaptchaGate from "@components\/article-captcha\/ArticleCaptchaGate\.astro";/,
	);
	assert.doesNotMatch(postsPage, /<ArticleCaptchaGate>/);
});

test("article captcha gate component exposes sitewide defaults and first-visit site copy", async () => {
	const gateComponent = await readRepoFile("src/components/article-captcha/ArticleCaptchaGate.astro");

	assert.match(gateComponent, /storageKey = "site-captcha:passed"/);
	assert.match(gateComponent, /backgroundImageUrl = "\/captcha\/preview\.jpg"/);
	assert.match(gateComponent, /fallbackBackgroundImageUrl = "\/captcha\/placeholder-background\.svg"/);
	assert.match(gateComponent, /localhost rotate captcha sandbox/);
	assert.match(gateComponent, /旋转拼图验证码/);
	assert.match(gateComponent, /拖动下方滑块，旋转中央圆片，让圆内图像与周围背景重新拼接到正确角度。/);
	assert.match(gateComponent, /当前主图按原始比例显示，不再强制拉伸。/);
	assert.match(gateComponent, /拖动滑块旋转中央圆片/);
	assert.match(gateComponent, /max="100"/);
	assert.match(gateComponent, /step="0\.01"/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*position:\s*fixed;/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*inset:\s*0;/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*overflow-y:\s*auto;/);
	assert.match(gateComponent, /\.article-captcha-card\s*\{[\s\S]*width:\s*min\(100%,\s*860px\);/);
	assert.match(gateComponent, /\.article-captcha-card\s*\{[\s\S]*max-height:\s*calc\(100vh\s*-\s*56px\);/);
	assert.match(gateComponent, /\.article-captcha-card\s*\{[\s\S]*overflow-y:\s*auto;/);
	assert.match(gateComponent, /\.article-captcha-slider\s*\{[\s\S]*height:\s*20px;/);
});

test("article captcha rotate logic matches the simulator challenge model", async () => {
	const logicModule = await import("../src/scripts/article-captcha/logic.mjs");
	const canvasSize = logicModule.resolveCanvasSize({
		sourceWidth: 864,
		sourceHeight: 864,
		maxCanvasWidth: 620,
	});

	assert.deepEqual(canvasSize, { canvasWidth: 620, canvasHeight: 620 });

	const circleRadius = logicModule.resolveCircleRadius({
		canvasWidth: 620,
		canvasHeight: 620,
		padding: 18,
		circleRadiusRatio: 0.18,
	});
	assert.equal(circleRadius, 112);

	const rngValues = [0.4, 0.25];
	let rngIndex = 0;
	const challenge = logicModule.createRotateChallenge({
		canvasWidth: 620,
		canvasHeight: 620,
		circleRadius,
		padding: 18,
		sliderMinValue: 0,
		sliderMaxValue: 100,
		minTravelTurns: 0.5,
		maxTravelTurns: 0.95,
		targetSliderPaddingRatio: 0.18,
		rng: () => rngValues[rngIndex++] ?? 0.5,
	});

	assert.deepEqual(challenge.circleCenter, { x: 310, y: 310 });
	assert.equal(challenge.circleRadius, 112);
	assert.equal(challenge.targetRotationDeg, 0);
	assert.equal(challenge.startSliderValue, 0);
	assert.ok(challenge.targetSliderValue >= 18);
	assert.ok(challenge.targetSliderValue <= 82);
	assert.ok(challenge.rotationSpanDeg >= 180);
	assert.ok(challenge.rotationSpanDeg <= 342);
	assert.equal(
		challenge.degreesPerSliderUnit,
		challenge.rotationSpanDeg / (challenge.sliderMaxValue - challenge.sliderMinValue),
	);
	assert.deepEqual(
		logicModule.createFreshCaptchaState({
			startRotationDeg: challenge.startRotationDeg,
			startSliderValue: challenge.startSliderValue,
		}),
		{
			currentRotationDeg: challenge.startRotationDeg,
			sliderValue: challenge.startSliderValue,
			startRotationDeg: challenge.startRotationDeg,
			startSliderValue: challenge.startSliderValue,
			isAnimating: false,
			isLocked: false,
			status: "idle",
		},
	);

	assert.equal(
		logicModule.sliderValueToRotation({
			sliderValue: challenge.targetSliderValue,
			challenge,
		}),
		challenge.targetRotationDeg,
	);
	assert.equal(
		logicModule.sliderValueToRotation({
			sliderValue: challenge.startSliderValue,
			challenge,
		}),
		challenge.startRotationDeg,
	);
});

test("article captcha rotate evaluation handles wraparound angles and shortest-path rebound math", async () => {
	const logicModule = await import("../src/scripts/article-captcha/logic.mjs");

	assert.deepEqual(
		logicModule.evaluateRotationAttempt({
			currentRotationDeg: 358,
			targetRotationDeg: 2,
			toleranceDeg: 6,
		}),
		{ success: true, deltaDeg: 4 },
	);
	assert.equal(
		logicModule.interpolateRotationDeg({
			fromDeg: 350,
			toDeg: 0,
			progress: 0.5,
		}),
		355,
	);
	assert.equal(
		logicModule.sliderValueToRotation({
			sliderValue: -10,
			challenge: {
				targetRotationDeg: 0,
				targetSliderValue: 50,
				sliderMinValue: 0,
				sliderMaxValue: 100,
				degreesPerSliderUnit: 2.4,
			},
		}),
		240,
	);
});

test("article captcha runtime keeps simulator-style slider behavior, session pass memory, and no challenge recorder", async () => {
	const runtime = await readRepoFile("src/scripts/article-captcha/index.ts");

	assert.match(runtime, /sessionStorage/);
	assert.doesNotMatch(runtime, /localStorage/);
	assert.match(runtime, /const storageKey = root\.dataset\.storageKey \?\? "site-captcha:passed";/);
	assert.match(runtime, /hooks\.on\("page:view", \(\) => initializeArticleCaptchaGates\(\)\)/);
	assert.match(runtime, /const SUCCESS_DISMISS_DELAY_MS = 500;/);
	assert.match(runtime, /await pause\(SUCCESS_DISMISS_DELAY_MS\);/);
	assert.match(runtime, /sliderMinValue:\s*0/);
	assert.match(runtime, /sliderMaxValue:\s*100/);
	assert.match(runtime, /minTravelTurns:\s*0\.5/);
	assert.match(runtime, /maxTravelTurns:\s*0\.95/);
	assert.match(runtime, /targetSliderPaddingRatio:\s*0\.18/);
	assert.doesNotMatch(runtime, /recordChallenge/);
	assert.doesNotMatch(runtime, /challengeRecordEndpoint/);
});

test("article captcha background assets are published from the blog public directory", async () => {
	await access(new URL("../public/captcha/preview.jpg", import.meta.url));
	await access(new URL("../public/captcha/placeholder-background.svg", import.meta.url));
});
