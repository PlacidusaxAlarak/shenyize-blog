import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { createReadStream } from "node:fs";
import { mkdir, readdir, stat, unlink, writeFile } from "node:fs/promises";
import sharp from "sharp";

const OPEN_IMAGES_VALIDATION_METADATA_URL =
	"https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv";
const OPEN_IMAGES_HUMAN_LABELS_URL =
	"https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv";
const OPEN_IMAGES_BBOX_LABELS_URL =
	"https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv";
const OPEN_IMAGES_CLASS_DESCRIPTIONS_URL =
	"https://storage.googleapis.com/openimages/v5/class-descriptions.csv";
const DEFAULT_FETCH_TIMEOUT_MS = 30_000;
const DEFAULT_FETCH_RETRIES = 2;

export const openImagesSampleTargetCount = 200;
export const openImagesPinnedSampleTargets = Object.freeze([
	Object.freeze({
		imageId: "00794645d77184eb",
		slug: "openimages-garage-door.jpg",
		subject: "garage-door",
		labels: Object.freeze(["building", "door", "window"]),
	}),
	Object.freeze({
		imageId: "0a3f577a327ca7cc",
		slug: "openimages-apartment-facade.jpg",
		subject: "apartment-facade",
		labels: Object.freeze(["building", "window"]),
	}),
	Object.freeze({
		imageId: "115ef722923602a8",
		slug: "openimages-townhouse-facade.jpg",
		subject: "townhouse-facade",
		labels: Object.freeze(["door", "window"]),
	}),
	Object.freeze({
		imageId: "0a556c8163b58fae",
		slug: "openimages-cafe-interior.jpg",
		subject: "cafe-interior",
		labels: Object.freeze(["building", "window"]),
	}),
	Object.freeze({
		imageId: "0b8ba050b1d83bb7",
		slug: "openimages-columned-facade.jpg",
		subject: "columned-facade",
		labels: Object.freeze(["building", "window"]),
	}),
	Object.freeze({
		imageId: "0af4ba8fd2a7e628",
		slug: "openimages-bathroom-sink.jpg",
		subject: "bathroom-sink",
		labels: Object.freeze(["bathroom"]),
	}),
	Object.freeze({
		imageId: "15f0965b63397a40",
		slug: "openimages-built-in-bedroom.jpg",
		subject: "built-in-bedroom",
		labels: Object.freeze(["bedroom"]),
	}),
	Object.freeze({
		imageId: "0651ced5fb5f7a21",
		slug: "openimages-living-room-sofa.jpg",
		subject: "living-room-sofa",
		labels: Object.freeze(["living-room"]),
	}),
	Object.freeze({
		imageId: "0598d112cfe889ea",
		slug: "openimages-crosswalk-street.jpg",
		subject: "crosswalk-street",
		labels: Object.freeze(["street"]),
	}),
	Object.freeze({
		imageId: "003e1e6baff436f7",
		slug: "openimages-garden-planters.jpg",
		subject: "garden-planters",
		labels: Object.freeze(["garden"]),
	}),
	Object.freeze({
		imageId: "04728d00324ff2af",
		slug: "openimages-turtle-beach.jpg",
		subject: "turtle-beach",
		labels: Object.freeze(["beach"]),
	}),
	Object.freeze({
		imageId: "00a72fa141918070",
		slug: "openimages-forest-stream.jpg",
		subject: "forest-stream",
		labels: Object.freeze(["forest", "river"]),
	}),
	Object.freeze({
		imageId: "035515479d72da78",
		slug: "openimages-lake-overlook.jpg",
		subject: "lake-overlook",
		labels: Object.freeze(["lake"]),
	}),
	Object.freeze({
		imageId: "2727d94d98934630",
		slug: "openimages-forest-waterfall.jpg",
		subject: "forest-waterfall",
		labels: Object.freeze(["waterfall"]),
	}),
	Object.freeze({
		imageId: "01472ac056cd787e",
		slug: "openimages-rocky-ocean.jpg",
		subject: "rocky-ocean",
		labels: Object.freeze(["ocean", "coast"]),
	}),
	Object.freeze({
		imageId: "0210e5bd37398d59",
		slug: "openimages-ice-sailboat.jpg",
		subject: "ice-sailboat",
		labels: Object.freeze(["boat"]),
	}),
	Object.freeze({
		imageId: "00fe64cf0eedb1cb",
		slug: "openimages-rustic-cabin.jpg",
		subject: "rustic-cabin",
		labels: Object.freeze(["house"]),
	}),
	Object.freeze({
		imageId: "04da2550440dd624",
		slug: "openimages-mountain-valley.jpg",
		subject: "mountain-valley",
		labels: Object.freeze(["valley"]),
	}),
	Object.freeze({
		imageId: "0cccffa267fce7bb",
		slug: "openimages-spiral-stairs.jpg",
		subject: "spiral-stairs",
		labels: Object.freeze(["stairs"]),
	}),
	Object.freeze({
		imageId: "02687c67bb4c5d55",
		slug: "openimages-bay-coast.jpg",
		subject: "bay-coast",
		labels: Object.freeze(["coast"]),
	}),
]);
export const openImagesSampleTargets = openImagesPinnedSampleTargets;
export const openImagesSampleThemeRules = Object.freeze([
	Object.freeze({
		theme: "architecture",
		matchMode: "any",
		positiveLabelNames: Object.freeze(["Building", "Door", "Window"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "bathroom",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Bathroom"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "bedroom",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Bedroom"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "living-room",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Living room"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "street",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Street"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "garden",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Garden"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "beach",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Beach"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "forest",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Forest"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "river",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["River"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "lake",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Lake"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "waterfall",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Waterfall"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "ocean",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Ocean"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "boat",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Boat"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "house",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["House"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "valley",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Valley"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "stairs",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Stairs"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
	Object.freeze({
		theme: "coast",
		matchMode: "all",
		positiveLabelNames: Object.freeze(["Coast"]),
		negativeLabelNames: Object.freeze(["Person"]),
	}),
]);

function isMissingFileError(error) {
	return Boolean(error && typeof error === "object" && "code" in error && error.code === "ENOENT");
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

function normalizeLabel(value) {
	return String(value).trim().toLowerCase().replace(/\s+/g, "-");
}

function parseCsvLine(line) {
	const values = [];
	let current = "";
	let isQuoted = false;

	for (let index = 0; index < line.length; index += 1) {
		const character = line[index];

		if (character === '"') {
			if (isQuoted && line[index + 1] === '"') {
				current += '"';
				index += 1;
				continue;
			}

			isQuoted = !isQuoted;
			continue;
		}

		if (character === "," && !isQuoted) {
			values.push(current);
			current = "";
			continue;
		}

		current += character;
	}

	values.push(current);
	return values;
}

function createPaths() {
	const sampleDir = path.resolve(process.cwd(), "public/openimages-sample");
	const cacheDir = path.resolve(process.cwd(), "scripts/cache/openimages");

	return {
		sampleDir,
		metadataPath: path.join(sampleDir, "metadata.json"),
		cacheDir,
		metadataCachePath: path.join(cacheDir, "validation-images-with-rotation.csv"),
		humanLabelsCachePath: path.join(cacheDir, "validation-annotations-human-imagelabels.csv"),
		bboxLabelsCachePath: path.join(cacheDir, "validation-annotations-bbox.csv"),
		classDescriptionsCachePath: path.join(cacheDir, "class-descriptions.csv"),
	};
}

async function ensureDirectories(paths) {
	await mkdir(paths.sampleDir, { recursive: true });
	await mkdir(paths.cacheDir, { recursive: true });
}

async function fetchWithTimeout(
	url,
	{ timeoutMs = DEFAULT_FETCH_TIMEOUT_MS, fetchImpl = fetch, parseResponse } = {},
) {
	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

	try {
		const response = await fetchImpl(url, {
			signal: controller.signal,
			redirect: "follow",
			headers: {
				"user-agent": "Mozilla/5.0",
			},
		});
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

async function ensureCacheFile(url, filePath) {
	if (await pathExists(filePath)) {
		return filePath;
	}

	const payload = await fetchWithRetry(url, {
		timeoutMs: 120_000,
		parseResponse: (response) => response.arrayBuffer(),
	});
	await writeFile(filePath, Buffer.from(payload));
	return filePath;
}

async function loadCsvMapByImageId(cachePath) {
	const recordMap = new Map();
	const metadataStream = createReadStream(cachePath, { encoding: "utf8" });
	const lineReader = readline.createInterface({
		input: metadataStream,
		crlfDelay: Infinity,
	});
	let headers;

	for await (const line of lineReader) {
		if (!headers) {
			headers = parseCsvLine(line);
			continue;
		}

		if (line.trim() === "") {
			continue;
		}

		const values = parseCsvLine(line);
		const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
		recordMap.set(record.ImageID, record);
	}

	return recordMap;
}

async function loadClassNameMaps(cachePath) {
	const nameToId = new Map();
	const idToName = new Map();
	const classStream = createReadStream(cachePath, { encoding: "utf8" });
	const lineReader = readline.createInterface({
		input: classStream,
		crlfDelay: Infinity,
	});

	for await (const line of lineReader) {
		if (line.trim() === "") {
			continue;
		}

		const [labelId, labelName] = parseCsvLine(line);
		nameToId.set(labelName, labelId);
		idToName.set(labelId, labelName);
	}

	return {
		nameToId,
		idToName,
	};
}

async function mergeLabelFileIntoMap(cachePath, imageLabelsById) {
	const labelsStream = createReadStream(cachePath, { encoding: "utf8" });
	const lineReader = readline.createInterface({
		input: labelsStream,
		crlfDelay: Infinity,
	});
	let headers;

	for await (const line of lineReader) {
		if (!headers) {
			headers = parseCsvLine(line);
			continue;
		}

		if (line.trim() === "") {
			continue;
		}

		const values = parseCsvLine(line);
		const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
		if ("Confidence" in record && record.Confidence !== "1" && record.Confidence !== "1.0") {
			continue;
		}

		const imageId = record.ImageID;
		if (!imageLabelsById.has(imageId)) {
			imageLabelsById.set(imageId, new Set());
		}

		imageLabelsById.get(imageId).add(record.LabelName);
	}
}

async function loadImageLabelsById({ humanLabelsCachePath, bboxLabelsCachePath }) {
	const imageLabelsById = new Map();
	await mergeLabelFileIntoMap(humanLabelsCachePath, imageLabelsById);
	await mergeLabelFileIntoMap(bboxLabelsCachePath, imageLabelsById);
	return imageLabelsById;
}

function resolveThemeRules({ themeRules, classNameToId }) {
	return themeRules.map((rule) => ({
		...rule,
		positiveLabelIds: rule.positiveLabelNames.map((labelName) => {
			const labelId = classNameToId.get(labelName);
			if (!labelId) {
				throw new Error(`Missing Open Images class id for label: ${labelName}`);
			}
			return labelId;
		}),
		negativeLabelIds: (rule.negativeLabelNames ?? []).map((labelName) => {
			const labelId = classNameToId.get(labelName);
			if (!labelId) {
				throw new Error(`Missing Open Images class id for label: ${labelName}`);
			}
			return labelId;
		}),
	}));
}

function matchesThemeRule({ imageLabelIds, themeRule }) {
	if (!imageLabelIds) {
		return false;
	}

	for (const labelId of themeRule.negativeLabelIds) {
		if (imageLabelIds.has(labelId)) {
			return false;
		}
	}

	if (themeRule.matchMode === "any") {
		return themeRule.positiveLabelIds.some((labelId) => imageLabelIds.has(labelId));
	}

	return themeRule.positiveLabelIds.every((labelId) => imageLabelIds.has(labelId));
}

function createCandidateTarget({ record, themeRule, imageLabelIds, idToName }) {
	const matchedLabels = themeRule.positiveLabelIds
		.filter((labelId) => imageLabelIds.has(labelId))
		.map((labelId) => normalizeLabel(idToName.get(labelId) ?? themeRule.theme));

	return {
		imageId: record.ImageID,
		slug: `openimages-${themeRule.theme}-${record.ImageID}.jpg`,
		subject: themeRule.theme,
		labels: matchedLabels.length > 0 ? matchedLabels : [themeRule.theme],
	};
}

function buildCandidateTargetsByTheme({
	metadataRecords,
	imageLabelsById,
	pinnedTargets,
	resolvedThemeRules,
	idToName,
}) {
	const pinnedTargetIds = new Set(pinnedTargets.map((target) => target.imageId));
	const candidateTargetsByTheme = new Map(
		resolvedThemeRules.map((themeRule) => [themeRule.theme, []]),
	);

	for (const record of metadataRecords.values()) {
		if (pinnedTargetIds.has(record.ImageID)) {
			continue;
		}

		if (!record.Thumbnail300KURL && !record.OriginalURL) {
			continue;
		}

		const imageLabelIds = imageLabelsById.get(record.ImageID);
		if (!imageLabelIds) {
			continue;
		}

		for (const themeRule of resolvedThemeRules) {
			if (!matchesThemeRule({ imageLabelIds, themeRule })) {
				continue;
			}

			candidateTargetsByTheme.get(themeRule.theme).push(
				createCandidateTarget({
					record,
					themeRule,
					imageLabelIds,
					idToName,
				}),
			);
		}
	}

	for (const [theme, candidates] of candidateTargetsByTheme) {
		candidateTargetsByTheme.set(
			theme,
			candidates.sort((first, second) => first.imageId.localeCompare(second.imageId)),
		);
	}

	return candidateTargetsByTheme;
}

export function selectOpenImagesSampleTargets({
	pinnedTargets,
	candidateTargetsByTheme,
	targetCount,
}) {
	const selectedTargets = [...pinnedTargets];
	const selectedImageIds = new Set(pinnedTargets.map((target) => target.imageId));
	const themeNames = [...candidateTargetsByTheme.keys()];
	const themeIndexes = new Map(themeNames.map((themeName) => [themeName, 0]));

	while (selectedTargets.length < targetCount) {
		let advanced = false;

		for (const themeName of themeNames) {
			if (selectedTargets.length >= targetCount) {
				break;
			}

			const candidates = candidateTargetsByTheme.get(themeName) ?? [];
			let candidateIndex = themeIndexes.get(themeName) ?? 0;

			while (
				candidateIndex < candidates.length &&
				selectedImageIds.has(candidates[candidateIndex].imageId)
			) {
				candidateIndex += 1;
			}

			if (candidateIndex >= candidates.length) {
				themeIndexes.set(themeName, candidateIndex);
				continue;
			}

			const nextTarget = candidates[candidateIndex];
			selectedTargets.push(nextTarget);
			selectedImageIds.add(nextTarget.imageId);
			themeIndexes.set(themeName, candidateIndex + 1);
			advanced = true;
		}

		if (!advanced) {
			break;
		}
	}

	if (selectedTargets.length < targetCount) {
		throw new Error(
			`Unable to resolve ${targetCount} Open Images targets; only found ${selectedTargets.length}.`,
		);
	}

	return selectedTargets;
}

function countUniqueCandidateTargets({ pinnedTargets, candidateTargetsByTheme }) {
	const uniqueImageIds = new Set(pinnedTargets.map((target) => target.imageId));

	for (const candidates of candidateTargetsByTheme.values()) {
		for (const candidate of candidates) {
			uniqueImageIds.add(candidate.imageId);
		}
	}

	return uniqueImageIds.size;
}

async function resolveOpenImagesSampleTargets(paths) {
	await Promise.all([
		ensureCacheFile(OPEN_IMAGES_VALIDATION_METADATA_URL, paths.metadataCachePath),
		ensureCacheFile(OPEN_IMAGES_HUMAN_LABELS_URL, paths.humanLabelsCachePath),
		ensureCacheFile(OPEN_IMAGES_BBOX_LABELS_URL, paths.bboxLabelsCachePath),
		ensureCacheFile(OPEN_IMAGES_CLASS_DESCRIPTIONS_URL, paths.classDescriptionsCachePath),
	]);

	const [metadataRecords, { nameToId, idToName }, imageLabelsById] = await Promise.all([
		loadCsvMapByImageId(paths.metadataCachePath),
		loadClassNameMaps(paths.classDescriptionsCachePath),
		loadImageLabelsById({
			humanLabelsCachePath: paths.humanLabelsCachePath,
			bboxLabelsCachePath: paths.bboxLabelsCachePath,
		}),
	]);
	const resolvedThemeRules = resolveThemeRules({
		themeRules: openImagesSampleThemeRules,
		classNameToId: nameToId,
	});
	const candidateTargetsByTheme = buildCandidateTargetsByTheme({
		metadataRecords,
		imageLabelsById,
		pinnedTargets: openImagesPinnedSampleTargets,
		resolvedThemeRules,
		idToName,
	});
	const maxAvailableTargetCount = countUniqueCandidateTargets({
		pinnedTargets: openImagesPinnedSampleTargets,
		candidateTargetsByTheme,
	});
	const orderedTargets = selectOpenImagesSampleTargets({
		pinnedTargets: openImagesPinnedSampleTargets,
		candidateTargetsByTheme,
		targetCount: maxAvailableTargetCount,
	});

	return {
		metadataRecords,
		orderedTargets,
	};
}

async function downloadImageBuffer(url) {
	return Buffer.from(
		await fetchWithRetry(url, {
			timeoutMs: 60_000,
			parseResponse: (response) => response.arrayBuffer(),
		}),
	);
}

async function downloadBestImageBuffer(record) {
	const candidateUrls = [record.OriginalURL, record.Thumbnail300KURL].filter(Boolean);
	let lastError;

	for (const candidateUrl of candidateUrls) {
		try {
			return {
				buffer: await downloadImageBuffer(candidateUrl),
				downloadedUrl: candidateUrl,
			};
		} catch (error) {
			lastError = error;
		}
	}

	throw lastError ?? new Error(`No downloadable image URL found for ${record.ImageID}`);
}

function getLicenseFamily(licenseUrl) {
	return typeof licenseUrl === "string" && /creativecommons\.org\/licenses\/by\//i.test(licenseUrl)
		? "CC-BY"
		: "other";
}

export function createOpenImagesSampleMetadataEntry({
	target,
	record,
	output,
	fetchedAt,
}) {
	return {
		id: `openimages-${record.ImageID}`,
		source: "openimages",
		imageId: record.ImageID,
		subset: record.Subset,
		title: record.Title,
		subject: target.subject,
		labels: [...target.labels],
		imageUrl: record.OriginalURL,
		thumbnailUrl: record.Thumbnail300KURL,
		localPath: output.localPath,
		landingUrl: record.OriginalLandingURL,
		license: record.License,
		licenseFamily: getLicenseFamily(record.License),
		author: record.Author,
		authorProfileUrl: record.AuthorProfileURL,
		width: output.width,
		height: output.height,
		rotation: Number.parseFloat(record.Rotation || "0") || 0,
		fetchedAt,
	};
}

async function writeProcessedImage({ record, outputFilePath }) {
	const { buffer } = await downloadBestImageBuffer(record);
	const image = sharp(buffer, { failOn: "none" }).rotate();
	const metadata = await image.metadata();
	const outputBuffer = await image.jpeg({ quality: 90, mozjpeg: true }).toBuffer();
	await writeFile(outputFilePath, outputBuffer);

	return {
		width: metadata.width ?? 0,
		height: metadata.height ?? 0,
	};
}

async function cleanupSampleDirectory({ sampleDir, retainedFileNames }) {
	const sampleEntries = await readdir(sampleDir, { withFileTypes: true });

	await Promise.all(
		sampleEntries
			.filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".jpg"))
			.filter((entry) => !retainedFileNames.has(entry.name))
			.map((entry) => unlink(path.join(sampleDir, entry.name))),
	);
}

export async function runOpenImagesSampleFetch() {
	const paths = createPaths();
	await ensureDirectories(paths);

	const { metadataRecords, orderedTargets } = await resolveOpenImagesSampleTargets(paths);
	const fetchedAt = new Date().toISOString();
	const metadataEntries = [];

	for (const target of orderedTargets) {
		if (metadataEntries.length >= openImagesSampleTargetCount) {
			break;
		}

		const record = metadataRecords.get(target.imageId);
		if (!record) {
			throw new Error(`Missing Open Images metadata for ${target.imageId}`);
		}

		const outputFilePath = path.join(paths.sampleDir, target.slug);
		const localPath = `/openimages-sample/${target.slug}`;
		let output;
		try {
			output = await writeProcessedImage({
				record,
				outputFilePath,
			});
		} catch (error) {
			console.warn(`Skipping Open Images sample ${target.imageId}: ${error}`);
			continue;
		}

		metadataEntries.push(
			createOpenImagesSampleMetadataEntry({
				target,
				record,
				output: {
					localPath,
					width: output.width,
					height: output.height,
				},
				fetchedAt,
			}),
		);
	}

	if (metadataEntries.length < openImagesSampleTargetCount) {
		throw new Error(
			`Open Images sample fetch only produced ${metadataEntries.length} images; expected ${openImagesSampleTargetCount}.`,
		);
	}

	await cleanupSampleDirectory({
		sampleDir: paths.sampleDir,
		retainedFileNames: new Set(metadataEntries.map((entry) => path.basename(entry.localPath))),
	});
	await writeFile(paths.metadataPath, `${JSON.stringify(metadataEntries, null, 2)}\n`, "utf8");

	return {
		metadataEntries,
		paths,
	};
}

const executedFilePath = process.argv[1] ? path.resolve(process.argv[1]) : null;
const currentFilePath = fileURLToPath(import.meta.url);

if (executedFilePath && currentFilePath === executedFilePath) {
	const result = await runOpenImagesSampleFetch();
	console.log(`Open Images sample images added: ${result.metadataEntries.length}`);
	for (const entry of result.metadataEntries) {
		console.log(`${entry.imageId} ${entry.localPath}`);
	}
}
