import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import sharp from "sharp";

import {
	isLikelyScenicCaptchaObject,
	isLikelySparseCaptchaPresentation,
	mergeManifestEntries,
} from "./fetch-captcha-backgrounds.mjs";

const CMA_API_BASE_URL = "https://openaccess-api.clevelandart.org/api";
const DEFAULT_FETCH_TIMEOUT_MS = 30_000;
const DEFAULT_FETCH_RETRIES = 3;
const DEFAULT_MIN_LONG_EDGE = 1000;
const DEFAULT_SAMPLE_LIMIT = 5;
const DEFAULT_SEARCH_LIMIT = 20;
const CONTENT_ANALYSIS_SAMPLE_SIZE = 160;
const CONTENT_ANALYSIS_BORDER_WIDTH = 8;
const CONTENT_ANALYSIS_DISTANCE_THRESHOLD = 26;

export const cmaSampleKeywords = Object.freeze([
	"landscape",
	"architecture",
	"bridge",
	"street",
	"garden",
]);
export const cmaPhotographPriorityKeywords = Object.freeze([
	"photograph landscape",
	"architectural photograph",
	"bridge photograph",
	"street photograph",
	"garden photograph",
	"architecture photograph",
	"city photograph",
]);

export function resolveCmaRemainingSlots({ manifestEntries, targetCount }) {
	if (!Array.isArray(manifestEntries) || !Number.isFinite(targetCount) || targetCount <= 0) {
		return 0;
	}

	const existingCmaCount = manifestEntries.filter((entry) => entry?.source === "cma").length;
	return Math.max(0, Math.floor(targetCount) - existingCmaCount);
}

export function resolveCmaSearchKeywords({ preferPhotographs = false } = {}) {
	return preferPhotographs ? cmaPhotographPriorityKeywords : cmaSampleKeywords;
}

const CMA_TYPE_REJECT_PATTERNS = [
	/\bsilver\b/,
	/\bceramic(s)?\b/,
	/\bglass\b/,
	/\bjewel(l)?ery\b/,
	/\bfurniture\b/,
	/\barmor\b/,
	/\bweapon(s)?\b/,
	/\bcoin(s)?\b/,
	/\bmedal(s)?\b/,
	/\bmanuscript(s)?\b/,
	/\bbook(s)?\b/,
	/\btextile(s)?\b/,
];

const CMA_TITLE_REJECT_PATTERNS = [
	/\bvirgin\b/,
	/\bmadonna\b/,
	/\bchild\b/,
	/\bsaint(s)?\b/,
	/\bchrist\b/,
	/\bcrucifixion\b/,
	/\bportrait(s)?\b/,
	/\bself-portrait\b/,
	/\bbust\b/,
];
const CMA_PHOTOGRAPH_SCENE_PATTERNS = [
	/\blandscape\b/,
	/\bstreet\b/,
	/\bbridge\b/,
	/\barchitecture\b/,
	/\barchitectural\b/,
	/\bbuilding(s)?\b/,
	/\bcity\b/,
	/\bgarden\b/,
	/\bpark\b/,
	/\bfountain\b/,
	/\btemple\b/,
	/\bcastle\b/,
	/\bchurch\b/,
	/\bcathedral\b/,
	/\bcourtyard\b/,
	/\bharbou?r\b/,
	/\briver\b/,
	/\bmountain\b/,
	/\bwood(ed)?\b/,
];

function sanitizeFileStem(value) {
	return String(value)
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 80);
}

function normalizeString(value) {
	return typeof value === "string" ? value.trim() : "";
}

function normalizeLowercaseTag(value) {
	const normalized = normalizeString(value).toLowerCase();
	return normalized.length > 0 ? normalized : null;
}

function uniqueStrings(values) {
	return [
		...new Set(
			values
				.filter((value) => typeof value === "string")
				.map((value) => value.trim())
				.filter(Boolean),
		),
	];
}

function isMissingFileError(error) {
	return Boolean(error && typeof error === "object" && "code" in error && error.code === "ENOENT");
}

async function readJsonFile(filePath) {
	return JSON.parse(await readFile(filePath, "utf8"));
}

async function readJsonFileIfExists(filePath) {
	try {
		return await readJsonFile(filePath);
	} catch (error) {
		if (isMissingFileError(error)) {
			return undefined;
		}

		throw error;
	}
}

async function writeJsonFile(filePath, value) {
	await mkdir(path.dirname(filePath), { recursive: true });
	await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function pathExists(filePath) {
	try {
		await stat(filePath);
		return true;
	} catch (error) {
		if (isMissingFileError(error)) {
			return false;
		}

		throw error;
	}
}

function createCmaSearchUrl({ keyword, skip = 0, limit = DEFAULT_SEARCH_LIMIT }) {
	const url = new URL(`${CMA_API_BASE_URL}/artworks/`);
	url.searchParams.set("q", keyword);
	url.searchParams.set("skip", String(skip));
	url.searchParams.set("limit", String(limit));
	return url;
}

function createCmaArtworkUrl(id) {
	return new URL(`${CMA_API_BASE_URL}/artworks/${id}/`);
}

async function fetchWithTimeout(
	url,
	{ timeoutMs = DEFAULT_FETCH_TIMEOUT_MS, fetchImpl = fetch, parseResponse } = {},
) {
	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

	try {
		const response = await fetchImpl(url, { signal: controller.signal });
		if (!response.ok) {
			throw new Error(`Unable to fetch ${url}: ${response.status} ${response.statusText}`);
		}

		return parseResponse ? parseResponse(response) : response;
	} finally {
		clearTimeout(timeoutId);
	}
}

async function fetchWithRetry(
	url,
	{
		timeoutMs = DEFAULT_FETCH_TIMEOUT_MS,
		retries = DEFAULT_FETCH_RETRIES,
		fetchImpl = fetch,
		parseResponse,
	} = {},
) {
	let lastError;

	for (let attempt = 0; attempt <= retries; attempt += 1) {
		try {
			return await fetchWithTimeout(url, {
				timeoutMs,
				fetchImpl,
				parseResponse,
			});
		} catch (error) {
			lastError = error;
			if (attempt === retries) {
				throw error;
			}
		}
	}

	throw lastError;
}

async function fetchJsonWithCache(url, cacheFilePath) {
	const cachedValue = await readJsonFileIfExists(cacheFilePath);
	if (cachedValue) {
		return cachedValue;
	}

	const payload = await fetchWithRetry(url, {
		parseResponse: (response) => response.json(),
	});
	await writeJsonFile(cacheFilePath, payload);
	return payload;
}

function createCmaPaths() {
	const outputDir = path.resolve(process.cwd(), "public/captcha/backgrounds");
	const cacheDir = path.resolve(process.cwd(), "scripts/cache/captcha");

	return {
		outputDir,
		manifestPath: path.join(outputDir, "manifest.json"),
		cmaDir: path.join(outputDir, "cma"),
		cacheSearchDir: path.join(cacheDir, "cma", "search"),
		cacheArtworkDir: path.join(cacheDir, "cma", "artworks"),
	};
}

async function ensureDirectories(paths) {
	await mkdir(paths.cmaDir, { recursive: true });
	await mkdir(paths.cacheSearchDir, { recursive: true });
	await mkdir(paths.cacheArtworkDir, { recursive: true });
}

function adaptCmaArtworkToScenicObject(artwork) {
	const department = normalizeString(artwork?.department);
	const collection = normalizeString(artwork?.collection);
	const type = normalizeString(artwork?.type);
	const technique = normalizeString(artwork?.technique);

	return {
		title: normalizeString(artwork?.title),
		objectName: type,
		classification: uniqueStrings([department, collection]).join(" "),
		department,
		medium: technique,
		tags: [],
	};
}

function getCmaArtworkText(artwork) {
	return [
		artwork?.title,
		artwork?.type,
		artwork?.department,
		artwork?.collection,
		artwork?.technique,
		...(Array.isArray(artwork?.alternate_titles) ? artwork.alternate_titles : []),
	]
		.filter((value) => typeof value === "string")
		.join(" ")
		.toLowerCase();
}

function getCmaImageCandidate(artwork) {
	const candidates = [artwork?.images?.print, artwork?.images?.web];
	for (const candidate of candidates) {
		if (!candidate?.url) {
			continue;
		}

		const width = Number(candidate.width ?? 0);
		const height = Number(candidate.height ?? 0);
		return {
			url: candidate.url,
			width,
			height,
		};
	}

	return null;
}

function countLongEdge(imageCandidate) {
	if (!imageCandidate) {
		return 0;
	}

	return Math.max(Number(imageCandidate.width) || 0, Number(imageCandidate.height) || 0);
}

export function isLikelyCmaSampleArtwork({ artwork, keyword }) {
	if (!artwork || artwork.share_license_status !== "CC0") {
		return false;
	}

	const imageCandidate = getCmaImageCandidate(artwork);
	if (!imageCandidate?.url || countLongEdge(imageCandidate) < DEFAULT_MIN_LONG_EDGE) {
		return false;
	}

	const artworkText = getCmaArtworkText(artwork);
	if (CMA_TYPE_REJECT_PATTERNS.some((pattern) => pattern.test(artworkText))) {
		return false;
	}

	if (CMA_TITLE_REJECT_PATTERNS.some((pattern) => pattern.test(artworkText))) {
		return false;
	}

	return isLikelyScenicCaptchaObject({
		object: adaptCmaArtworkToScenicObject(artwork),
		keyword,
	});
}

function isLikelyCmaPhotographArtwork({ artwork, keyword }) {
	if (!artwork || artwork.share_license_status !== "CC0") {
		return false;
	}

	const imageCandidate = getCmaImageCandidate(artwork);
	if (!imageCandidate?.url || countLongEdge(imageCandidate) < DEFAULT_MIN_LONG_EDGE) {
		return false;
	}

	const artworkText = `${getCmaArtworkText(artwork)} ${normalizeString(keyword).toLowerCase()}`;
	if (normalizeString(artwork?.type).toLowerCase() !== "photograph") {
		return false;
	}

	if (CMA_TITLE_REJECT_PATTERNS.some((pattern) => pattern.test(artworkText))) {
		return false;
	}

	return CMA_PHOTOGRAPH_SCENE_PATTERNS.some((pattern) => pattern.test(artworkText));
}

function deriveCmaTags({ artwork, keyword }) {
	return uniqueStrings(
		[keyword, artwork?.type, artwork?.department, artwork?.collection]
			.map(normalizeLowercaseTag)
			.filter(Boolean),
	);
}

export function scoreCmaCandidateForSort({
	artwork,
	keyword,
	preferPhotographs = false,
}) {
	const artworkText = `${getCmaArtworkText(artwork)} ${normalizeString(keyword).toLowerCase()}`;
	let score = 0;

	if (preferPhotographs) {
		if (normalizeString(artwork?.type).toLowerCase() === "photograph") {
			score += 10;
		}

		if (normalizeString(artwork?.department).toLowerCase() === "photography") {
			score += 6;
		}

		for (const pattern of CMA_PHOTOGRAPH_SCENE_PATTERNS) {
			if (pattern.test(artworkText)) {
				score += 2;
			}
		}

		return score;
	}

	for (const pattern of [/\bbridge\b/, /\bstreet\b/, /\barchitecture\b/, /\blandscape\b/, /\bgarden\b/]) {
		if (pattern.test(artworkText)) {
			score += 1;
		}
	}

	return score;
}

export function createCmaManifestEntry({ artwork, output, keyword, fetchedAt }) {
	const imageCandidate = getCmaImageCandidate(artwork);

	return {
		id: `cma-${artwork.id}`,
		source: "cma",
		objectId: artwork.id,
		title: normalizeString(artwork.title),
		imageUrl: imageCandidate?.url ?? "",
		localPath: output.localPath,
		objectUrl: normalizeString(artwork.url),
		license: "CC0",
		width: output.width,
		height: output.height,
		tags: deriveCmaTags({ artwork, keyword }),
		fetchedAt,
	};
}

function getColorDistance(first, second) {
	return Math.sqrt(
		(first[0] - second[0]) ** 2 +
			(first[1] - second[1]) ** 2 +
			(first[2] - second[2]) ** 2,
	);
}

async function measureImageContentMetrics(image) {
	const { data, info } = await image
		.clone()
		.resize({
			width: CONTENT_ANALYSIS_SAMPLE_SIZE,
			height: CONTENT_ANALYSIS_SAMPLE_SIZE,
			fit: "inside",
		})
		.removeAlpha()
		.raw()
		.toBuffer({ resolveWithObject: true });
	const { width, height, channels } = info;
	const borderWidth = Math.max(
		1,
		Math.min(CONTENT_ANALYSIS_BORDER_WIDTH, Math.floor(Math.min(width, height) / 4)),
	);
	const borderPixels = [];

	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			if (
				x < borderWidth ||
				y < borderWidth ||
				x >= width - borderWidth ||
				y >= height - borderWidth
			) {
				const index = (y * width + x) * channels;
				borderPixels.push([data[index], data[index + 1], data[index + 2]]);
			}
		}
	}

	const backgroundColor = [0, 1, 2].map(
		(channelIndex) =>
			borderPixels.reduce((sum, pixel) => sum + pixel[channelIndex], 0) /
			Math.max(borderPixels.length, 1),
	);

	let minX = width;
	let minY = height;
	let maxX = -1;
	let maxY = -1;
	let contentPixelCount = 0;

	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			const index = (y * width + x) * channels;
			const pixel = [data[index], data[index + 1], data[index + 2]];
			if (getColorDistance(pixel, backgroundColor) <= CONTENT_ANALYSIS_DISTANCE_THRESHOLD) {
				continue;
			}

			contentPixelCount += 1;
			minX = Math.min(minX, x);
			minY = Math.min(minY, y);
			maxX = Math.max(maxX, x);
			maxY = Math.max(maxY, y);
		}
	}

	const totalPixelCount = width * height;
	const bboxAreaRatio =
		maxX >= minX && maxY >= minY
			? ((maxX - minX + 1) * (maxY - minY + 1)) / totalPixelCount
			: 0;

	return {
		contentRatio: contentPixelCount / Math.max(totalPixelCount, 1),
		bboxAreaRatio,
	};
}

async function downloadImageBuffer(imageUrl) {
	return Buffer.from(
		await fetchWithRetry(imageUrl, {
			timeoutMs: 45_000,
			parseResponse: (response) => response.arrayBuffer(),
		}),
	);
}

async function processImageBuffer({ imageBuffer, outputFilePath, artwork }) {
	const image = sharp(imageBuffer, { failOn: "none" }).rotate();
	const metadata = await image.metadata();
	const width = metadata.width ?? 0;
	const height = metadata.height ?? 0;
	if (Math.max(width, height) < DEFAULT_MIN_LONG_EDGE) {
		return {
			accepted: false,
			reason: "low_resolution",
			width,
			height,
		};
	}

	const contentMetrics = await measureImageContentMetrics(image);
	if (
		isLikelySparseCaptchaPresentation({
			object: {
				objectName: artwork?.type,
				classification: artwork?.department,
			},
			metrics: contentMetrics,
		})
	) {
		return {
			accepted: false,
			reason: "incomplete_composition",
			width,
			height,
		};
	}

	const outputBuffer = await image.jpeg({ quality: 88, mozjpeg: true }).toBuffer();
	await mkdir(path.dirname(outputFilePath), { recursive: true });
	await writeFile(outputFilePath, outputBuffer);

	return {
		accepted: true,
		width,
		height,
	};
}

async function readExistingOutputMetadata(outputFilePath) {
	const metadata = await sharp(outputFilePath).metadata();
	return {
		width: metadata.width ?? 0,
		height: metadata.height ?? 0,
	};
}

async function loadExistingManifestEntries(manifestPath) {
	const entries = await readJsonFileIfExists(manifestPath);
	return Array.isArray(entries) ? entries : [];
}

async function fetchCmaSearchPage({ keyword, skip, paths }) {
	const cacheFilePath = path.join(
		paths.cacheSearchDir,
		`${sanitizeFileStem(`${keyword}-${skip}`)}.json`,
	);
	return fetchJsonWithCache(createCmaSearchUrl({ keyword, skip }), cacheFilePath);
}

async function fetchCmaArtwork({ id, paths }) {
	const cacheFilePath = path.join(paths.cacheArtworkDir, `${id}.json`);
	const payload = await fetchJsonWithCache(createCmaArtworkUrl(id), cacheFilePath);
	return payload?.data ?? payload;
}

async function collectCmaCandidateIds({ paths, preferPhotographs = false }) {
	const candidates = [];
	const seen = new Set();
	const searchKeywords = resolveCmaSearchKeywords({ preferPhotographs });
	const searchOffsets = preferPhotographs
		? [0, DEFAULT_SEARCH_LIMIT, DEFAULT_SEARCH_LIMIT * 2, DEFAULT_SEARCH_LIMIT * 3, DEFAULT_SEARCH_LIMIT * 4]
		: [0, DEFAULT_SEARCH_LIMIT, DEFAULT_SEARCH_LIMIT * 2];

	for (const keyword of searchKeywords) {
		for (const skip of searchOffsets) {
			const payload = await fetchCmaSearchPage({ keyword, skip, paths });
			const artworks = Array.isArray(payload?.data) ? payload.data : [];
			if (artworks.length === 0) {
				break;
			}

			for (const artwork of artworks) {
				if (!artwork?.id || seen.has(artwork.id)) {
					continue;
				}

				seen.add(artwork.id);
				candidates.push({ keyword, id: artwork.id });
			}
		}
	}

	return candidates;
}

export async function collectCmaArtworkDetails({
	candidates,
	loadArtwork,
	preferPhotographs = false,
}) {
	const detailedCandidates = [];

	for (const candidate of candidates) {
		let artwork;
		try {
			artwork = await loadArtwork(candidate);
		} catch {
			continue;
		}

		const accepted = preferPhotographs
			? isLikelyCmaPhotographArtwork({ artwork, keyword: candidate.keyword })
			: isLikelyCmaSampleArtwork({ artwork, keyword: candidate.keyword });
		if (!artwork || !accepted) {
			continue;
		}

		detailedCandidates.push({
			keyword: candidate.keyword,
			artwork,
		});
	}

	return detailedCandidates;
}

function sortCmaCandidates(entries, { preferPhotographs = false } = {}) {
	return [...entries].sort((first, second) => {
		const firstScore = scoreCmaCandidateForSort({
			artwork: first.artwork,
			keyword: first.keyword,
			preferPhotographs,
		});
		const secondScore = scoreCmaCandidateForSort({
			artwork: second.artwork,
			keyword: second.keyword,
			preferPhotographs,
		});
		return secondScore - firstScore;
	});
}

export async function runCmaSampleFetch({
	targetCount = DEFAULT_SAMPLE_LIMIT,
	preferPhotographs = false,
} = {}) {
	const paths = createCmaPaths();
	await ensureDirectories(paths);

	const existingManifestEntries = await loadExistingManifestEntries(paths.manifestPath);
	const manifestEntries = mergeManifestEntries(existingManifestEntries, []);
	const manifestEntryMap = new Map(manifestEntries.map((entry) => [entry.id, entry]));
	const remainingSlots = resolveCmaRemainingSlots({
		manifestEntries,
		targetCount,
	});

	if (remainingSlots === 0) {
		return {
			addedEntries: [],
			manifestEntries,
			paths,
		};
	}

	const candidates = await collectCmaCandidateIds({ paths, preferPhotographs });
	const detailedCandidates = await collectCmaArtworkDetails({
		candidates,
		preferPhotographs,
		loadArtwork: (candidate) => fetchCmaArtwork({ id: candidate.id, paths }),
	});

	const fetchedAt = new Date().toISOString();
	const addedEntries = [];

	for (const candidate of sortCmaCandidates(detailedCandidates, { preferPhotographs })) {
		if (addedEntries.length >= remainingSlots) {
			break;
		}

		const artwork = candidate.artwork;
		const manifestId = `cma-${artwork.id}`;
		if (manifestEntryMap.has(manifestId)) {
			continue;
		}

		const fileName = `${manifestId}.jpg`;
		const outputFilePath = path.join(paths.cmaDir, fileName);
		const localPath = `/captcha/backgrounds/cma/${fileName}`;

		if (await pathExists(outputFilePath)) {
			const existingOutput = await readExistingOutputMetadata(outputFilePath);
			const manifestEntry = createCmaManifestEntry({
				artwork,
				output: {
					localPath,
					width: existingOutput.width,
					height: existingOutput.height,
				},
				keyword: candidate.keyword,
				fetchedAt,
			});
			manifestEntries.push(manifestEntry);
			manifestEntryMap.set(manifestEntry.id, manifestEntry);
			addedEntries.push(manifestEntry);
			continue;
		}

		const imageCandidate = getCmaImageCandidate(artwork);
		const imageBuffer = await downloadImageBuffer(imageCandidate.url);
		const processedImage = await processImageBuffer({
			imageBuffer,
			outputFilePath,
			artwork,
		});
		if (!processedImage.accepted) {
			continue;
		}

		const manifestEntry = createCmaManifestEntry({
			artwork,
			output: {
				localPath,
				width: processedImage.width,
				height: processedImage.height,
			},
			keyword: candidate.keyword,
			fetchedAt,
		});
		manifestEntries.push(manifestEntry);
		manifestEntryMap.set(manifestEntry.id, manifestEntry);
		addedEntries.push(manifestEntry);
	}

	const mergedEntries = mergeManifestEntries([], manifestEntries);
	await writeJsonFile(paths.manifestPath, mergedEntries);

	return {
		addedEntries,
		manifestEntries: mergedEntries,
		paths,
	};
}

const executedFilePath = process.argv[1] ? path.resolve(process.argv[1]) : null;
const currentFilePath = fileURLToPath(import.meta.url);

if (executedFilePath && currentFilePath === executedFilePath) {
	const requestedTargetCount = Number(process.argv[2]);
	const preferPhotographs = process.argv.includes("--prefer-photographs");
	const result = await runCmaSampleFetch({
		targetCount:
			Number.isFinite(requestedTargetCount) && requestedTargetCount > 0
				? requestedTargetCount
				: DEFAULT_SAMPLE_LIMIT,
		preferPhotographs,
	});
	console.log(`CMA sample images added: ${result.addedEntries.length}`);
	for (const entry of result.addedEntries) {
		console.log(`${entry.id} ${entry.title}`);
	}
}
