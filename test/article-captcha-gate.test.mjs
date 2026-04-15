import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);

async function readRepoFile(relativePath) {
	const filePath = new URL(relativePath, repoRoot);
	return readFile(filePath, "utf8");
}

function normalizeRotationDeg(value) {
	const normalized = value % 360;
	return normalized < 0 ? normalized + 360 : normalized;
}

function rectanglesOverlap(first, second) {
	return !(
		first.right <= second.left ||
		first.left >= second.right ||
		first.bottom <= second.top ||
		first.top >= second.bottom
	);
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

test("article captcha gate component exposes sitewide defaults without hint copy and keeps the slider visible in the viewport", async () => {
	const gateComponent = await readRepoFile("src/components/article-captcha/ArticleCaptchaGate.astro");

	assert.match(gateComponent, /storageKey = "site-captcha:passed"/);
	assert.match(gateComponent, /backgroundImageUrl = "\/captcha\/preview\.jpg"/);
	assert.match(gateComponent, /fallbackBackgroundImageUrl = "\/captcha\/placeholder-background\.svg"/);
	assert.match(gateComponent, /安全验证/);
	assert.match(gateComponent, /拖动滑块完成验证/);
	assert.doesNotMatch(gateComponent, /localhost rotate captcha sandbox/);
	assert.doesNotMatch(gateComponent, /旋转拼图验证码/);
	assert.doesNotMatch(gateComponent, /拖动下方滑块，旋转中央圆片，让圆内图像与周围背景重新拼接到正确角度。/);
	assert.doesNotMatch(gateComponent, /当前主图按原始比例显示，不再强制拉伸。/);
	assert.doesNotMatch(gateComponent, /每道题的正确位置和旋转灵敏度都会随机变化/);
	assert.match(gateComponent, /max="100"/);
	assert.match(gateComponent, /step="0\.01"/);
	assert.match(gateComponent, /data-article-captcha-controls/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*position:\s*fixed;/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*inset:\s*0;/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*padding:\s*clamp\(12px,\s*2\.8dvh,\s*24px\)\s+18px;/);
	assert.match(gateComponent, /\.article-captcha-card\s*\{[\s\S]*width:\s*min\(100%,\s*860px\);/);
	assert.match(gateComponent, /\.article-captcha-controls\s*\{[\s\S]*position:\s*absolute;/);
	assert.doesNotMatch(gateComponent, /data-article-captcha-refresh/);
	assert.doesNotMatch(gateComponent, /\.article-captcha-refresh/);
	assert.match(gateComponent, /\.article-captcha-overlay\s*\{[\s\S]*--article-captcha-overlay-padding:\s*clamp\(12px,\s*2\.8dvh,\s*24px\);/);
	assert.match(
		gateComponent,
		/\.article-captcha-card\s*\{[\s\S]*max-height:\s*calc\(100dvh\s*-\s*\(var\(--article-captcha-overlay-padding\)\s*\*\s*2\)\);/,
	);
	assert.match(gateComponent, /\.article-captcha-card\s*\{[\s\S]*display:\s*grid;/);
	assert.match(
		gateComponent,
		/\.article-captcha-canvas-frame\s*\{[\s\S]*width:\s*min\(100%,\s*var\(--article-captcha-canvas-limit,\s*700px\)\);/,
	);
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

	const rngValues = [0.4, 0.25, 0.25, 0.9, 0.2];
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
	assert.ok(challenge.sensitivityScale >= 0.65);
	assert.ok(challenge.sensitivityScale <= 1.35);
	assert.equal(Number(challenge.sensitivityScale.toFixed(2)), 1.28);
	assert.equal(challenge.rotationDirection, -1);
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
	assert.equal(
		Number(
			logicModule
				.sliderValueToRotation({
					sliderValue: challenge.targetSliderValue + 10,
					challenge,
				})
				.toFixed(3),
		),
		Number(
			normalizeRotationDeg(
				challenge.degreesPerSliderUnit *
					10 *
					challenge.sensitivityScale *
					challenge.rotationDirection,
			).toFixed(3),
		),
	);

	let reverseRngIndex = 0;
	const reverseDirectionChallenge = logicModule.createRotateChallenge({
		canvasWidth: 620,
		canvasHeight: 620,
		circleRadius,
		padding: 18,
		sliderMinValue: 0,
		sliderMaxValue: 100,
		minTravelTurns: 0.5,
		maxTravelTurns: 0.95,
		targetSliderPaddingRatio: 0.18,
		rng: () => [0.4, 0.25, 0.25, 0.9, 0.8][reverseRngIndex++] ?? 0.5,
	});

	assert.equal(reverseDirectionChallenge.rotationDirection, 1);
	assert.notEqual(challenge.startRotationDeg, reverseDirectionChallenge.startRotationDeg);
	assert.equal(
		Number(challenge.startRotationDeg.toFixed(3)),
		Number(
			logicModule
				.sliderValueToRotation({
					sliderValue: challenge.startSliderValue,
					challenge,
				})
				.toFixed(3),
		),
	);
	assert.equal(
		Number(reverseDirectionChallenge.startRotationDeg.toFixed(3)),
		Number(
			logicModule
				.sliderValueToRotation({
					sliderValue: reverseDirectionChallenge.startSliderValue,
					challenge: reverseDirectionChallenge,
				})
				.toFixed(3),
		),
	);
});

test("article captcha rotate logic can start from either slider edge and float controls away from the image", async () => {
	const logicModule = await import("../src/scripts/article-captcha/logic.mjs");

	let leftStartIndex = 0;
	const leftStartChallenge = logicModule.createRotateChallenge({
		canvasWidth: 620,
		canvasHeight: 620,
		circleRadius: 112,
		padding: 18,
		sliderMinValue: 0,
		sliderMaxValue: 100,
		rng: () => [0.35, 0.2, 0.45, 0.6, 0.3][leftStartIndex++] ?? 0.5,
	});
	assert.equal(leftStartChallenge.startSliderValue, 0);

	let rightStartIndex = 0;
	const rightStartChallenge = logicModule.createRotateChallenge({
		canvasWidth: 620,
		canvasHeight: 620,
		circleRadius: 112,
		padding: 18,
		sliderMinValue: 0,
		sliderMaxValue: 100,
		rng: () => [0.35, 0.8, 0.45, 0.6, 0.3][rightStartIndex++] ?? 0.5,
	});
	assert.equal(rightStartChallenge.startSliderValue, 100);

	const floatingPosition = logicModule.resolveFloatingPanelPosition({
		viewportWidth: 1280,
		viewportHeight: 860,
		panelWidth: 320,
		panelHeight: 144,
		blockedRect: {
			left: 400,
			top: 160,
			right: 860,
			bottom: 620,
		},
		padding: 24,
		gap: 18,
		rng: () => 0.75,
	});
	const floatingRect = {
		left: floatingPosition.left,
		top: floatingPosition.top,
		right: floatingPosition.left + 320,
		bottom: floatingPosition.top + 144,
	};

	assert.ok(floatingRect.left >= 24);
	assert.ok(floatingRect.top >= 24);
	assert.ok(floatingRect.right <= 1280 - 24);
	assert.ok(floatingRect.bottom <= 860 - 24);
	assert.equal(
		rectanglesOverlap(floatingRect, {
			left: 400,
			top: 160,
			right: 860,
			bottom: 620,
		}),
		false,
	);
});

test("article captcha visible canvas limit grows the image while keeping the card inside the viewport", async () => {
	const logicModule = await import("../src/scripts/article-captcha/logic.mjs");

	assert.equal(
		logicModule.resolveVisibleCanvasLimit({
			viewportWidth: 1280,
			viewportHeight: 900,
			overlayPadding: 24,
			cardPaddingX: 44,
			cardPaddingY: 44,
			headerHeight: 88,
			contentGap: 18,
			framePaddingX: 24,
			framePaddingY: 24,
			maxCanvasWidth: 760,
		}),
		678,
	);

	assert.equal(
		logicModule.resolveVisibleCanvasLimit({
			viewportWidth: 820,
			viewportHeight: 720,
			overlayPadding: 18,
			cardPaddingX: 36,
			cardPaddingY: 36,
			headerHeight: 84,
			contentGap: 16,
			framePaddingX: 20,
			framePaddingY: 20,
			maxCanvasWidth: 760,
		}),
		528,
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
	assert.match(runtime, /resolveFloatingPanelPosition/);
	assert.match(runtime, /resolveVisibleCanvasLimit/);
	assert.match(runtime, /sliderMinValue:\s*0/);
	assert.match(runtime, /sliderMaxValue:\s*100/);
	assert.match(runtime, /minTravelTurns:\s*0\.5/);
	assert.match(runtime, /maxTravelTurns:\s*0\.95/);
	assert.match(runtime, /targetSliderPaddingRatio:\s*0\.18/);
	assert.match(runtime, /focus\(\{\s*preventScroll:\s*true\s*\}\)/);
	assert.doesNotMatch(runtime, /refreshButton/);
	assert.doesNotMatch(runtime, /CAPTCHA_REFRESHED_TEXT/);
	assert.doesNotMatch(runtime, /createChallenge\("refresh"\)/);
	assert.doesNotMatch(runtime, /每道题的正确位置和旋转灵敏度都会随机变化/);
	assert.doesNotMatch(runtime, /角度误差/);
	assert.doesNotMatch(runtime, /recordChallenge/);
	assert.doesNotMatch(runtime, /challengeRecordEndpoint/);
});

test("article captcha background assets are published from the blog public directory", async () => {
	await access(new URL("../public/captcha/preview.jpg", import.meta.url));
	await access(new URL("../public/captcha/placeholder-background.svg", import.meta.url));
});
