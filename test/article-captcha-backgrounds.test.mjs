import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import sharp from "sharp";

const backgroundPoolModule = () =>
	import("../src/components/article-captcha/background-image-pool.mjs");
const logicModule = () => import("../src/scripts/article-captcha/logic.mjs");

test("article captcha discovers supported local images from the openimages sample directory", async () => {
	const tempPublicDir = await mkdtemp(path.join(os.tmpdir(), "captcha-background-pool-"));
	const backgroundsDir = path.join(tempPublicDir, "openimages-sample");
	const nestedDir = path.join(backgroundsDir, "nested");

	await mkdir(nestedDir, { recursive: true });
	await sharp({
		create: {
			width: 120,
			height: 120,
			channels: 3,
			background: { r: 214, g: 197, b: 169 },
		},
	})
		.composite([
			{
				input: Buffer.from(`
					<svg width="120" height="120" xmlns="http://www.w3.org/2000/svg">
						<line x1="18" y1="26" x2="102" y2="94" stroke="#5b371d" stroke-width="18" stroke-linecap="round" />
						<circle cx="82" cy="46" r="18" fill="#8d5a30" />
					</svg>
				`),
			},
		])
		.jpeg()
		.toFile(path.join(backgroundsDir, "scene-a.jpg"));
	await writeFile(path.join(backgroundsDir, "ignore.txt"), "x");
	await sharp({
		create: {
			width: 120,
			height: 120,
			channels: 3,
			background: { r: 141, g: 103, b: 68 },
		},
	})
		.composite([
			{
				input: Buffer.from(`
					<svg width="120" height="120" xmlns="http://www.w3.org/2000/svg">
						<path d="M18 88 L54 30 L96 82" fill="none" stroke="#f1d2a2" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" />
						<rect x="56" y="40" width="24" height="22" rx="4" fill="#f7e3bf" />
					</svg>
				`),
			},
		])
		.webp()
		.toFile(path.join(nestedDir, "scene-b.webp"));

	const { getArticleCaptchaBackgroundImageUrls } = await backgroundPoolModule();
	const imageUrls = await getArticleCaptchaBackgroundImageUrls({
		publicDir: tempPublicDir,
		fallbackUrls: ["/openimages-sample/fallback.jpg"],
	});

	assert.deepEqual(imageUrls, [
		"/openimages-sample/nested/scene-b.webp",
		"/openimages-sample/scene-a.jpg",
	]);
});

test("article captcha random image selection avoids repeating the previous image when alternatives exist", async () => {
	const { pickRandomBackgroundImageUrl } = await logicModule();

	const selected = pickRandomBackgroundImageUrl({
		imageUrls: [
			"/openimages-sample/scene-seed.jpg",
			"/openimages-sample/scene-a.jpg",
			"/openimages-sample/scene-b.jpg",
		],
		previousImageUrl: "/openimages-sample/scene-seed.jpg",
		rng: () => 0,
	});

	assert.equal(selected, "/openimages-sample/scene-a.jpg");
	assert.equal(
		pickRandomBackgroundImageUrl({
			imageUrls: ["/openimages-sample/scene-seed.jpg"],
			previousImageUrl: "/openimages-sample/scene-seed.jpg",
			rng: () => 0.5,
		}),
		"/openimages-sample/scene-seed.jpg",
	);
});

test("article captcha filters out openimages sample files whose center area does not provide a visible rotation cue", async () => {
	const tempPublicDir = await mkdtemp(path.join(os.tmpdir(), "captcha-background-quality-"));
	const backgroundsDir = path.join(tempPublicDir, "openimages-sample");
	const blankImagePath = path.join(backgroundsDir, "blank-center.jpg");
	const directionalImagePath = path.join(backgroundsDir, "directional-center.jpg");

	await mkdir(backgroundsDir, { recursive: true });
	await sharp({
		create: {
			width: 800,
			height: 1200,
			channels: 3,
			background: { r: 232, g: 216, b: 188 },
		},
	})
		.composite([
			{
				input: Buffer.from(`
					<svg width="800" height="1200" xmlns="http://www.w3.org/2000/svg">
						<rect x="150" y="920" width="500" height="180" rx="28" fill="#4f3722" opacity="0.92" />
					</svg>
				`),
			},
		])
		.jpeg()
		.toFile(blankImagePath);
	await sharp({
		create: {
			width: 800,
			height: 1200,
			channels: 3,
			background: { r: 232, g: 216, b: 188 },
		},
	})
		.composite([
			{
				input: Buffer.from(`
					<svg width="800" height="1200" xmlns="http://www.w3.org/2000/svg">
						<line x1="180" y1="250" x2="620" y2="650" stroke="#2f2115" stroke-width="72" stroke-linecap="round" />
						<circle cx="520" cy="420" r="74" fill="#87532b" />
					</svg>
				`),
			},
		])
		.jpeg()
		.toFile(directionalImagePath);
	await writeFile(
		path.join(backgroundsDir, "manifest.json"),
		`${JSON.stringify(
			[
				{
					id: "blank-center",
					localPath: "/openimages-sample/blank-center.jpg",
					width: 800,
					height: 1200,
				},
				{
					id: "directional-center",
					localPath: "/openimages-sample/directional-center.jpg",
					width: 800,
					height: 1200,
				},
			],
			null,
			2,
		)}\n`,
	);

	const { getArticleCaptchaBackgroundImageUrls } = await backgroundPoolModule();
	const imageUrls = await getArticleCaptchaBackgroundImageUrls({
		publicDir: tempPublicDir,
		fallbackUrls: ["/openimages-sample/fallback.jpg"],
	});

	assert.deepEqual(imageUrls, [
		"/openimages-sample/directional-center.jpg",
	]);
});

test("article captcha falls back only when the openimages sample directory has no usable images", async () => {
	const tempPublicDir = await mkdtemp(path.join(os.tmpdir(), "captcha-background-fallback-"));

	const { getArticleCaptchaBackgroundImageUrls } = await backgroundPoolModule();
	const imageUrls = await getArticleCaptchaBackgroundImageUrls({
		publicDir: tempPublicDir,
		fallbackUrls: ["/openimages-sample/fallback.jpg"],
	});

	assert.deepEqual(imageUrls, ["/openimages-sample/fallback.jpg"]);
});
