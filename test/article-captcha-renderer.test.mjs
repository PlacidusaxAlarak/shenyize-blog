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

test("article captcha surfaces the primary image load failure when it hangs past the timeout", async () => {
	const { loadBackgroundImage } = await rendererModule();
	const primaryImage = createFakeImage(() => {});
	await assert.rejects(
		() =>
			loadBackgroundImage("/captcha/missing.svg", {
				timeoutMs: 10,
				imageFactory() {
					return primaryImage;
				},
			}),
		/Timed out loading image: \/captcha\/missing\.svg/,
	);
});

test("article captcha keeps the primary image when it is slow but still finishes within the default timeout", async () => {
	const { loadBackgroundImage } = await rendererModule();
	const primaryImage = createFakeImage((_, image) => {
		setTimeout(() => {
			image.onload?.();
		}, 3500);
	});
	const result = await loadBackgroundImage("/captcha/slow.jpg", {
		imageFactory() {
			return primaryImage;
		},
	});

	assert.equal(result, primaryImage);
});

test("article captcha falls back to the next background image when the first choice times out", async () => {
	const { loadBackgroundImageFromSources } = await rendererModule();
	const imageStates = new Map();

	const result = await loadBackgroundImageFromSources(
		["/openimages-sample/first.jpg", "/openimages-sample/second.jpg"],
		{
			timeoutMs: 20,
			imageFactory() {
				return createFakeImage((value, image) => {
					imageStates.set(value, image);
					if (value === "/openimages-sample/second.jpg") {
						setTimeout(() => {
							image.onload?.();
						}, 0);
					}
				});
			},
		},
	);

	assert.equal(result.source, "/openimages-sample/second.jpg");
	assert.equal(result.image, imageStates.get("/openimages-sample/second.jpg"));
});
