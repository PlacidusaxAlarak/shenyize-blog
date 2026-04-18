import path from "node:path";
import { readdir } from "node:fs/promises";
import sharp from "sharp";

const SUPPORTED_IMAGE_EXTENSIONS = new Set([
	".avif",
	".gif",
	".jpeg",
	".jpg",
	".png",
	".svg",
	".webp",
]);
const CENTER_ANALYSIS_SAMPLE_SIZE = 240;
const CENTER_ANALYSIS_RADIUS_RATIO = 0.18;
const MIN_CENTER_VARIANCE = 120;
const MIN_ROTATION_CUE_DIFF = 12;
const backgroundImagePoolCache = new Map();

function normalizeUrl(value) {
	if (typeof value !== "string") {
		return null;
	}

	const normalized = value.trim();
	return normalized.length > 0 ? normalized : null;
}

function createBackgroundCandidate({ absolutePath, rootDirectory, urlBasePath }) {
	const relativePath = path.relative(rootDirectory, absolutePath).split(path.sep).join("/");

	return {
		absolutePath,
		url: `${urlBasePath}/${relativePath}`,
	};
}

async function collectBackgroundImageCandidates(directory, rootDirectory, urlBasePath) {
	let entries;
	try {
		entries = await readdir(directory, { withFileTypes: true });
	} catch (error) {
		if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
			return [];
		}

		throw error;
	}

	const imageCandidates = [];
	for (const entry of entries) {
		const absolutePath = path.join(directory, entry.name);
		if (entry.isDirectory()) {
			imageCandidates.push(
				...(await collectBackgroundImageCandidates(absolutePath, rootDirectory, urlBasePath)),
			);
			continue;
		}

		if (!entry.isFile()) {
			continue;
		}

		const extension = path.extname(entry.name).toLowerCase();
		if (!SUPPORTED_IMAGE_EXTENSIONS.has(extension)) {
			continue;
		}

		imageCandidates.push(createBackgroundCandidate({ absolutePath, rootDirectory, urlBasePath }));
	}

	return imageCandidates.sort((first, second) => first.url.localeCompare(second.url));
}

function computePixelVariance(grayscaleValues) {
	if (grayscaleValues.length === 0) {
		return 0;
	}

	const mean = grayscaleValues.reduce((sum, value) => sum + value, 0) / grayscaleValues.length;
	return grayscaleValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / grayscaleValues.length;
}

function sampleGrayscaleValue({ data, width, height, channels, x, y }) {
	const safeX = Math.max(0, Math.min(width - 1, Math.round(x)));
	const safeY = Math.max(0, Math.min(height - 1, Math.round(y)));
	const index = (safeY * width + safeX) * channels;

	return (data[index] + data[index + 1] + data[index + 2]) / 3;
}

function measureCenterRotationCue({ data, width, height, channels }) {
	const centerX = (width - 1) / 2;
	const centerY = (height - 1) / 2;
	const radius = Math.min(width, height) * CENTER_ANALYSIS_RADIUS_RATIO;
	const radiusSquared = radius * radius;
	const grayscaleValues = [];
	let rotationDifferenceSum = 0;
	let sampledPixelCount = 0;

	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			const deltaX = x - centerX;
			const deltaY = y - centerY;
			if (deltaX * deltaX + deltaY * deltaY > radiusSquared) {
				continue;
			}

			const baseValue = sampleGrayscaleValue({
				data,
				width,
				height,
				channels,
				x,
				y,
			});
			const rotatedValue = sampleGrayscaleValue({
				data,
				width,
				height,
				channels,
				x: centerX - deltaY,
				y: centerY + deltaX,
			});

			grayscaleValues.push(baseValue);
			rotationDifferenceSum += Math.abs(baseValue - rotatedValue);
			sampledPixelCount += 1;
		}
	}

	return {
		variance: computePixelVariance(grayscaleValues),
		rotationCueDifference:
			sampledPixelCount > 0 ? rotationDifferenceSum / sampledPixelCount : 0,
	};
}

async function hasVisibleCenterRotationCue(absolutePath) {
	const { data, info } = await sharp(absolutePath)
		.rotate()
		.resize({
			width: CENTER_ANALYSIS_SAMPLE_SIZE,
			height: CENTER_ANALYSIS_SAMPLE_SIZE,
			fit: "inside",
		})
		.removeAlpha()
		.raw()
		.toBuffer({ resolveWithObject: true });
	const { variance, rotationCueDifference } = measureCenterRotationCue({
		data,
		width: info.width,
		height: info.height,
		channels: info.channels,
	});

	return variance >= MIN_CENTER_VARIANCE && rotationCueDifference >= MIN_ROTATION_CUE_DIFF;
}

async function resolveUsableBackgroundImageUrls(publicDir) {
	const backgroundDirectory = path.join(publicDir, "openimages-sample");
	const discoveredImageCandidates = await collectBackgroundImageCandidates(
		backgroundDirectory,
		backgroundDirectory,
		"/openimages-sample",
	);
	const cueVisibilityResults = await Promise.all(
		discoveredImageCandidates.map(async (candidate) => {
			try {
				return {
					url: candidate.url,
					hasVisibleCue: await hasVisibleCenterRotationCue(candidate.absolutePath),
				};
			} catch {
				return {
					url: candidate.url,
					hasVisibleCue: false,
				};
			}
		}),
	);

	return cueVisibilityResults
		.filter((candidate) => candidate.hasVisibleCue)
		.map((candidate) => candidate.url);
}

export async function getArticleCaptchaBackgroundImageUrls({
	publicDir = path.resolve(process.cwd(), "public"),
	fallbackUrls = [],
} = {}) {
	const resolvedPublicDir = path.resolve(publicDir);
	let usableImageUrlsPromise = backgroundImagePoolCache.get(resolvedPublicDir);

	if (!usableImageUrlsPromise) {
		usableImageUrlsPromise = resolveUsableBackgroundImageUrls(resolvedPublicDir).catch((error) => {
			backgroundImagePoolCache.delete(resolvedPublicDir);
			throw error;
		});
		backgroundImagePoolCache.set(resolvedPublicDir, usableImageUrlsPromise);
	}
	const discoveredImageUrls = await usableImageUrlsPromise;

	if (discoveredImageUrls.length > 0) {
		return discoveredImageUrls;
	}

	return [...new Set(fallbackUrls.map(normalizeUrl).filter(Boolean))];
}
