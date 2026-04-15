import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import sharp from "sharp";

const MET_API_BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1";
const DEFAULT_METADATA_CONCURRENCY = 10;
const DEFAULT_DOWNLOAD_CONCURRENCY = 8;
const DEFAULT_FETCH_TIMEOUT_MS = 30_000;
const DEFAULT_FETCH_RETRIES = 3;
const DEFAULT_SEARCH_CANDIDATE_MULTIPLIER = 3;

function sanitizeFileStem(value) {
	return String(value)
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 80);
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

function createCounterMap() {
	return {
		candidatesFound: 0,
		detailRecordsFetched: 0,
		duplicatesSkipped: 0,
		imagesDownloaded: 0,
		reusedExistingFiles: 0,
		filtered: {},
	};
}

function incrementFiltered(stats, reason) {
	stats.filtered[reason] = (stats.filtered[reason] ?? 0) + 1;
}

function createMetSearchUrl(keyword) {
	const url = new URL(`${MET_API_BASE_URL}/search`);
	url.searchParams.set("q", keyword);
	url.searchParams.set("hasImages", "true");
	return url;
}

function createMetObjectUrl(objectId) {
	return new URL(`${MET_API_BASE_URL}/objects/${objectId}`);
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

export async function fetchWithRetry(
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

async function searchMetObjects({ keyword, cacheDir, maxPerKeyword }) {
	const cacheFilePath = path.join(
		cacheDir,
		"met",
		"search",
		`${sanitizeFileStem(keyword)}.json`,
	);
	const payload = await fetchJsonWithCache(createMetSearchUrl(keyword), cacheFilePath);
	const objectIds = Array.isArray(payload.objectIDs) ? payload.objectIDs : [];

	return objectIds.slice(
		0,
		Math.max(maxPerKeyword * DEFAULT_SEARCH_CANDIDATE_MULTIPLIER, maxPerKeyword),
	);
}

async function getMetObjectRecord({ objectId, cacheDir }) {
	const cacheFilePath = path.join(cacheDir, "met", "objects", `${objectId}.json`);
	return fetchJsonWithCache(createMetObjectUrl(objectId), cacheFilePath);
}

function getMetImageUrl(object) {
	if (typeof object?.primaryImage === "string" && object.primaryImage.trim() !== "") {
		return object.primaryImage.trim();
	}

	return null;
}

function normalizeTagTerm(value) {
	return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function getObjectTagTerms(object) {
	return Array.isArray(object?.tags)
		? object.tags.map((tag) => normalizeTagTerm(tag?.term)).filter(Boolean)
		: [];
}

function deriveObjectTags({ object, keyword }) {
	return uniqueStrings([
		normalizeTagTerm(keyword),
		normalizeTagTerm(object?.objectName),
		normalizeTagTerm(object?.classification),
		...getObjectTagTerms(object),
	]);
}

function getObjectSearchText({ object, keyword }) {
	return [
		keyword,
		object?.title,
		object?.objectName,
		object?.classification,
		object?.department,
		object?.medium,
		...getObjectTagTerms(object),
	]
		.filter((value) => typeof value === "string")
		.join(" ")
		.toLowerCase();
}

function getObjectMetadataText(object) {
	return [
		object?.title,
		object?.objectName,
		object?.classification,
		object?.department,
		object?.medium,
		...getObjectTagTerms(object),
	]
		.filter((value) => typeof value === "string")
		.join(" ")
		.toLowerCase();
}

function getObjectScenicText(object) {
	return [object?.title, ...getObjectTagTerms(object)]
		.filter((value) => typeof value === "string")
		.join(" ")
		.toLowerCase();
}

const SCENIC_STRONG_PATTERNS = [
	/\blandscape(s)?\b/,
	/\bseascape(s)?\b/,
	/\bcityscape(s)?\b/,
	/\bmountain(s)?\b/,
	/\bforest(s)?\b/,
	/\briver(s)?\b/,
	/\blake(s)?\b/,
	/\bvalley\b/,
	/\bwaterfall(s)?\b/,
	/\bcountryside\b/,
	/\bvillage(s)?\b/,
	/\bcoast(al)?\b/,
	/\bshore(s)?\b/,
	/\bharbou?r(s)?\b/,
	/\bbay(s)?\b/,
	/\bcanal(s)?\b/,
	/\bgarden(s)?\b/,
	/\bpark(s)?\b/,
	/\bbridge(s)?\b/,
	/\bstreet scene(s)?\b/,
	/\bstreet(s)?\b/,
	/\bavenue(s)?\b/,
	/\broad(s)?\b/,
	/\blane(s)?\b/,
	/\bview(s)?\b/,
	/\bpanorama(s)?\b/,
	/\bruins?\b/,
	/\bcastle(s)?\b/,
	/\btemple(s)?\b/,
	/\bpagoda(s)?\b/,
	/\bpalace(s)?\b/,
	/\bocean(s)?\b/,
	/\bwave(s)?\b/,
	/\brock(s)?\b/,
	/\bisland(s)?\b/,
	/\bcave(s)?\b/,
	/\bgrotto(es)?\b/,
	/\bbuilding(s)?\b/,
	/\bhouse(s)?\b/,
	/\bcourtyard(s)?\b/,
];
const SCENIC_SUPPORT_PATTERNS = [
	/\btrees\b/,
	/\bhill(s)?\b/,
	/\bfield(s)?\b/,
	/\bpath(s)?\b/,
	/\broad(s)?\b/,
	/\bmeadow(s)?\b/,
	/\bcliff(s)?\b/,
	/\bwaterfront\b/,
	/\bshoreline\b/,
	/\barchitecture\b/,
	/\barchitectural\b/,
	/\bbuildings\b/,
	/\bhouses\b/,
	/\bboats\b/,
	/\binterior(s)?\b/,
	/\bcolumns\b/,
	/\boffice building\b/,
	/\bold town\b/,
];
const SCENIC_REJECT_PATTERNS = [
	/\bcoin(s)?\b/,
	/\bmedal(s)?\b/,
	/\btextile(s)?\b/,
	/\bgarment(s)?\b/,
	/\bbracelet(s)?\b/,
	/\bnecklace(s)?\b/,
	/\bring(s)?\b/,
	/\bfragment(s)?\b/,
	/\bmanuscript(s)?\b/,
	/\bbook(s)?\b/,
	/\barmor\b/,
	/\bweapon(s)?\b/,
	/\bsword(s)?\b/,
	/\bportrait(s)?\b/,
	/\bself-portrait\b/,
	/\bbust(-length)?\b/,
	/\bstudy of (a|an|the) (man|woman|boy|girl|child)\b/,
	/\bstatue(s)?\b/,
	/\bstatuette(s)?\b/,
	/\bfigurine(s)?\b/,
	/\bmadonna\b/,
	/\bsaint(s)?\b/,
	/\bbuddha\b/,
	/\bbodhisattva\b/,
	/\bgoddess(es)?\b/,
	/\bdeit(y|ies)\b/,
	/\bvase(s)?\b/,
	/\bjug(s)?\b/,
	/\bplate(s)?\b/,
	/\btray(s)?\b/,
	/\bbowl(s)?\b/,
	/\bteabowl(s)?\b/,
	/\bclock\b/,
	/\bwatch\b/,
	/\bdesk\b/,
	/\bfurniture\b/,
	/\brelief(s)?\b/,
	/\bsampler(s)?\b/,
	/\bcalligraphy\b/,
	/\bpin(s)?\b/,
];

function countPatternMatches(text, patterns) {
	let score = 0;
	for (const pattern of patterns) {
		if (pattern.test(text)) {
			score += 1;
		}
	}
	return score;
}

export function isLikelyScenicCaptchaObject({ object, keyword: _keyword }) {
	const metadataText = getObjectMetadataText(object);
	if (countPatternMatches(metadataText, SCENIC_REJECT_PATTERNS) > 0) {
		return false;
	}

	const scenicText = getObjectScenicText(object) || metadataText;
	const strongMatches = countPatternMatches(scenicText, SCENIC_STRONG_PATTERNS);
	const supportMatches = countPatternMatches(metadataText, SCENIC_SUPPORT_PATTERNS);

	return strongMatches > 0 && strongMatches * 3 + supportMatches >= 3;
}

function scoreLandscapePriority({ object, keyword }) {
	if (!isLikelyScenicCaptchaObject({ object, keyword })) {
		return -1;
	}

	const metadataText = getObjectMetadataText(object);
	const scenicText = getObjectScenicText(object) || metadataText;

	return (
		countPatternMatches(scenicText, SCENIC_STRONG_PATTERNS) * 3 +
		countPatternMatches(metadataText, SCENIC_SUPPORT_PATTERNS)
	);
}

async function downloadImageBuffer(imageUrl) {
	return Buffer.from(
		await fetchWithRetry(imageUrl, {
			timeoutMs: 45_000,
			parseResponse: (response) => response.arrayBuffer(),
		}),
	);
}

async function processImageBuffer({ imageBuffer, outputFilePath, minLongEdge }) {
	const image = sharp(imageBuffer, { failOn: "none" }).rotate();
	const metadata = await image.metadata();
	const width = metadata.width ?? 0;
	const height = metadata.height ?? 0;
	if (Math.max(width, height) < minLongEdge) {
		return {
			accepted: false,
			reason: "low_resolution",
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

export function createManifestEntry({ object, output, keyword, fetchedAt }) {
	return {
		id: `met-${object.objectID}`,
		source: "met",
		objectId: object.objectID,
		title: object.title ?? "",
		imageUrl: object.primaryImage,
		localPath: output.localPath,
		objectUrl: object.objectURL ?? "",
		license: object.isPublicDomain === false ? "restricted" : "CC0",
		width: output.width,
		height: output.height,
		tags: deriveObjectTags({ object, keyword }),
		fetchedAt,
	};
}

export function mergeManifestEntries(existingEntries, newEntries) {
	const entryMap = new Map();

	for (const entry of [...existingEntries, ...newEntries]) {
		if (!entry || typeof entry !== "object" || !entry.id) {
			continue;
		}

		if (entryMap.has(entry.id)) {
			continue;
		}

		entryMap.set(entry.id, entry);
	}

	return [...entryMap.values()].sort((first, second) => first.objectId - second.objectId);
}

async function mapWithConcurrency(items, concurrency, mapper) {
	const results = [];
	let currentIndex = 0;

	const workers = Array.from(
		{ length: Math.max(1, Math.min(concurrency, items.length)) },
		async () => {
			while (currentIndex < items.length) {
				const index = currentIndex++;
				results[index] = await mapper(items[index], index);
			}
		},
	);

	await Promise.all(workers);
	return results;
}

async function loadConfig() {
	const configFilePath = path.resolve(
		process.cwd(),
		"scripts/fetch-captcha-backgrounds.config.json",
	);
	return readJsonFile(configFilePath);
}

function createOutputPaths(config) {
	const outputDir = path.resolve(process.cwd(), config.outputDir);
	return {
		outputDir,
		manifestPath: path.join(outputDir, "manifest.json"),
		metDir: path.join(outputDir, "met"),
		cacheDir: path.resolve(process.cwd(), config.cacheDir),
	};
}

async function ensureDirectories(paths) {
	await mkdir(paths.metDir, { recursive: true });
	await mkdir(paths.cacheDir, { recursive: true });
}

async function loadExistingManifestEntries(manifestPath) {
	const entries = await readJsonFileIfExists(manifestPath);
	return Array.isArray(entries) ? entries : [];
}

async function collectCandidateIds({ config, paths, stats }) {
	const candidateIds = [];
	const seenIds = new Set();

	for (const keyword of config.keywords) {
		const keywordIds = await searchMetObjects({
			keyword,
			cacheDir: paths.cacheDir,
			maxPerKeyword: config.maxPerKeyword,
		});

		stats.candidatesFound += keywordIds.length;
		for (const objectId of keywordIds) {
			if (seenIds.has(objectId)) {
				stats.duplicatesSkipped += 1;
				continue;
			}

			seenIds.add(objectId);
			candidateIds.push({ keyword, objectId });
		}
	}

	return candidateIds;
}

async function fetchCandidateObjectDetails({ candidates, paths, stats }) {
	const detailedCandidates = await mapWithConcurrency(
		candidates,
		DEFAULT_METADATA_CONCURRENCY,
		async (candidate) => {
			try {
				const object = await getMetObjectRecord({
					objectId: candidate.objectId,
					cacheDir: paths.cacheDir,
				});
				stats.detailRecordsFetched += 1;
				return {
					...candidate,
					object,
					priority: scoreLandscapePriority({
						object,
						keyword: candidate.keyword,
					}),
				};
			} catch (error) {
				incrementFiltered(stats, "detail_fetch_failed");
				return {
					...candidate,
					object: null,
					priority: -1,
					error,
				};
			}
		},
	);

	return detailedCandidates.sort((first, second) => second.priority - first.priority);
}

async function downloadMetBackgrounds({
	config,
	candidates,
	paths,
	stats,
	existingManifestEntries,
}) {
	const fetchedAt = new Date().toISOString();
	const manifestEntries = mergeManifestEntries(existingManifestEntries, []);
	const manifestEntryMap = new Map(manifestEntries.map((entry) => [entry.id, entry]));
	let currentIndex = 0;
	let manifestWriteQueue = Promise.resolve();

	const persistManifestEntries = async () => {
		const snapshot = mergeManifestEntries([], manifestEntries);
		manifestWriteQueue = manifestWriteQueue.then(() =>
			writeJsonFile(paths.manifestPath, snapshot),
		);
		await manifestWriteQueue;
	};

	const registerManifestEntry = async (
		manifestEntry,
		{ reusedExistingFile = false } = {},
	) => {
		if (manifestEntryMap.has(manifestEntry.id) || manifestEntries.length >= config.targetCount) {
			return false;
		}

		manifestEntries.push(manifestEntry);
		manifestEntryMap.set(manifestEntry.id, manifestEntry);
		if (reusedExistingFile) {
			stats.reusedExistingFiles += 1;
		} else {
			stats.imagesDownloaded += 1;
		}
		await persistManifestEntries();
		return true;
	};

	const workers = Array.from(
		{ length: Math.max(1, Math.min(DEFAULT_DOWNLOAD_CONCURRENCY, candidates.length)) },
		async () => {
			while (currentIndex < candidates.length) {
				if (manifestEntries.length >= config.targetCount) {
					return;
				}

				const candidate = candidates[currentIndex++];
				const object = candidate.object;
				if (!object || typeof object !== "object") {
					continue;
				}

				if (!object.isPublicDomain) {
					incrementFiltered(stats, "not_public_domain");
					continue;
				}

				if (!isLikelyScenicCaptchaObject({ object, keyword: candidate.keyword })) {
					incrementFiltered(stats, "low_relevance");
					continue;
				}

				const imageUrl = getMetImageUrl(object);
				if (!imageUrl) {
					incrementFiltered(stats, "missing_image");
					continue;
				}

				const manifestId = `met-${object.objectID}`;
				if (manifestEntryMap.has(manifestId)) {
					continue;
				}

				const fileName = `${manifestId}.jpg`;
				const outputFilePath = path.join(paths.metDir, fileName);
				const localPath = `/captcha/backgrounds/met/${fileName}`;

				try {
					if (await pathExists(outputFilePath)) {
						const existingOutput = await readExistingOutputMetadata(outputFilePath);
						await registerManifestEntry(
							createManifestEntry({
								object,
								output: {
									localPath,
									width: existingOutput.width,
									height: existingOutput.height,
								},
								keyword: candidate.keyword,
								fetchedAt,
							}),
							{ reusedExistingFile: true },
						);
						continue;
					}

					const imageBuffer = await downloadImageBuffer(imageUrl);
					const processedImage = await processImageBuffer({
						imageBuffer,
						outputFilePath,
						minLongEdge: config.minLongEdge,
					});

					if (!processedImage.accepted) {
						incrementFiltered(stats, processedImage.reason);
						continue;
					}

					await registerManifestEntry(
						createManifestEntry({
							object,
							output: {
								localPath,
								width: processedImage.width,
								height: processedImage.height,
							},
							keyword: candidate.keyword,
							fetchedAt,
						}),
					);
				} catch (error) {
					incrementFiltered(stats, "image_download_failed");
					console.error(`Image download failed for ${object.objectID}`, error);
				}
			}
		},
	);

	await Promise.all(workers);
	await manifestWriteQueue;

	return mergeManifestEntries([], manifestEntries);
}

function printSummary({ config, stats, manifestEntries, paths }) {
	console.log("Captcha background crawl summary");
	console.log(`Keywords processed: ${config.keywords.length}`);
	console.log(`Candidate IDs found: ${stats.candidatesFound}`);
	console.log(`Detail records fetched: ${stats.detailRecordsFetched}`);
	console.log(`Images downloaded: ${stats.imagesDownloaded}`);
	console.log(`Existing files reused: ${stats.reusedExistingFiles}`);
	console.log(`Duplicates skipped: ${stats.duplicatesSkipped}`);
	console.log(`Manifest entries: ${manifestEntries.length}`);
	console.log(`Manifest path: ${paths.manifestPath}`);

	for (const [reason, count] of Object.entries(stats.filtered).sort((a, b) =>
		a[0].localeCompare(b[0]),
	)) {
		console.log(`Filtered ${reason}: ${count}`);
	}
}

export async function runCaptchaBackgroundFetch() {
	const config = await loadConfig();
	const paths = createOutputPaths(config);
	const stats = createCounterMap();
	await ensureDirectories(paths);

	const existingManifestEntries = await loadExistingManifestEntries(paths.manifestPath);
	const candidates = await collectCandidateIds({ config, paths, stats });
	const detailedCandidates = await fetchCandidateObjectDetails({
		candidates,
		paths,
		stats,
	});
	const manifestEntries = await downloadMetBackgrounds({
		config,
		candidates: detailedCandidates,
		paths,
		stats,
		existingManifestEntries,
	});

	await writeJsonFile(paths.manifestPath, manifestEntries);
	printSummary({ config, stats, manifestEntries, paths });

	return {
		config,
		paths,
		stats,
		manifestEntries,
	};
}

const executedFilePath = process.argv[1] ? path.resolve(process.argv[1]) : null;
const currentFilePath = fileURLToPath(import.meta.url);

if (executedFilePath && currentFilePath === executedFilePath) {
	await runCaptchaBackgroundFetch();
}
