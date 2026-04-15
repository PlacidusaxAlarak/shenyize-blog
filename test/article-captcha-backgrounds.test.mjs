import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const backgroundPoolModule = () =>
	import("../src/components/article-captcha/background-image-pool.mjs");
const logicModule = () => import("../src/scripts/article-captcha/logic.mjs");

test("article captcha discovers supported local images from the public backgrounds directory", async () => {
	const tempPublicDir = await mkdtemp(path.join(os.tmpdir(), "captcha-background-pool-"));
	const backgroundsDir = path.join(tempPublicDir, "captcha", "backgrounds");
	const nestedDir = path.join(backgroundsDir, "nested");

	await mkdir(nestedDir, { recursive: true });
	await writeFile(path.join(backgroundsDir, "scene-a.jpg"), "a");
	await writeFile(path.join(backgroundsDir, "ignore.txt"), "x");
	await writeFile(path.join(nestedDir, "scene-b.webp"), "b");

	const { getArticleCaptchaBackgroundImageUrls } = await backgroundPoolModule();
	const imageUrls = await getArticleCaptchaBackgroundImageUrls({
		publicDir: tempPublicDir,
		fallbackUrls: ["/captcha/preview.jpg", "/captcha/demo-background.svg"],
	});

	assert.deepEqual(imageUrls, [
		"/captcha/preview.jpg",
		"/captcha/demo-background.svg",
		"/captcha/backgrounds/nested/scene-b.webp",
		"/captcha/backgrounds/scene-a.jpg",
	]);
});

test("article captcha random image selection avoids repeating the previous image when alternatives exist", async () => {
	const { pickRandomBackgroundImageUrl } = await logicModule();

	const selected = pickRandomBackgroundImageUrl({
		imageUrls: [
			"/captcha/preview.jpg",
			"/captcha/demo-background.svg",
			"/captcha/backgrounds/scene-a.jpg",
		],
		previousImageUrl: "/captcha/preview.jpg",
		rng: () => 0,
	});

	assert.equal(selected, "/captcha/demo-background.svg");
	assert.equal(
		pickRandomBackgroundImageUrl({
			imageUrls: ["/captcha/preview.jpg"],
			previousImageUrl: "/captcha/preview.jpg",
			rng: () => 0.5,
		}),
		"/captcha/preview.jpg",
	);
});
