import assert from "node:assert/strict";
import test from "node:test";

const rendererModule = () => import("../src/scripts/article-captcha/renderer.mjs");

function createFakeImage(onAssignSrc) {
	return {
		onload: null,
		onerror: null,
		set src(value) {
			this._src = value;
			onAssignSrc?.(value, this);
		},
		get src() {
			return this._src;
		},
	};
}

test("article captcha falls back when the primary image load hangs past the timeout", async () => {
	const { loadBackgroundImage } = await rendererModule();
	const primaryImage = createFakeImage(() => {});
	const fallbackImage = createFakeImage((_, image) => {
		setTimeout(() => {
			image.onload?.();
		}, 0);
	});
	let imageIndex = 0;

	const result = await loadBackgroundImage("/captcha/missing.svg", "/captcha/fallback.svg", {
		timeoutMs: 10,
		imageFactory() {
			return imageIndex++ === 0 ? primaryImage : fallbackImage;
		},
	});

	assert.equal(result.usedFallback, true);
	assert.equal(result.image, fallbackImage);
});

test("article captcha keeps the primary image when it is slow but still finishes within the default timeout", async () => {
	const { loadBackgroundImage } = await rendererModule();
	const primaryImage = createFakeImage((_, image) => {
		setTimeout(() => {
			image.onload?.();
		}, 3500);
	});
	const fallbackImage = createFakeImage(() => {});
	let imageIndex = 0;

	const result = await loadBackgroundImage("/captcha/slow.jpg", "/captcha/fallback.svg", {
		imageFactory() {
			return imageIndex++ === 0 ? primaryImage : fallbackImage;
		},
	});

	assert.equal(result.usedFallback, false);
	assert.equal(result.image, primaryImage);
});
