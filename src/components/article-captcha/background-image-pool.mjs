import path from "node:path";
import { readdir } from "node:fs/promises";

const SUPPORTED_IMAGE_EXTENSIONS = new Set([
	".avif",
	".gif",
	".jpeg",
	".jpg",
	".png",
	".svg",
	".webp",
]);

function normalizeUrl(value) {
	if (typeof value !== "string") {
		return null;
	}

	const normalized = value.trim();
	return normalized.length > 0 ? normalized : null;
}

async function collectBackgroundImageUrls(directory, rootDirectory) {
	let entries;
	try {
		entries = await readdir(directory, { withFileTypes: true });
	} catch (error) {
		if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
			return [];
		}

		throw error;
	}

	const imageUrls = [];
	for (const entry of entries) {
		const absolutePath = path.join(directory, entry.name);
		if (entry.isDirectory()) {
			imageUrls.push(...(await collectBackgroundImageUrls(absolutePath, rootDirectory)));
			continue;
		}

		if (!entry.isFile()) {
			continue;
		}

		const extension = path.extname(entry.name).toLowerCase();
		if (!SUPPORTED_IMAGE_EXTENSIONS.has(extension)) {
			continue;
		}

		const relativePath = path.relative(rootDirectory, absolutePath).split(path.sep).join("/");
		imageUrls.push(`/captcha/backgrounds/${relativePath}`);
	}

	return imageUrls;
}

export async function getArticleCaptchaBackgroundImageUrls({
	publicDir = path.resolve(process.cwd(), "public"),
	fallbackUrls = [],
} = {}) {
	const backgroundDirectory = path.join(publicDir, "captcha", "backgrounds");
	const discoveredImageUrls = await collectBackgroundImageUrls(
		backgroundDirectory,
		backgroundDirectory,
	);

	return [...new Set([...fallbackUrls.map(normalizeUrl), ...discoveredImageUrls].filter(Boolean))];
}
