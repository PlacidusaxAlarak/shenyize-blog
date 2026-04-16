import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);

async function readRepoFile(relativePath) {
	return readFile(new URL(relativePath, repoRoot), "utf8");
}

test("captcha background fetch config targets the local captcha backgrounds directory", async () => {
	await access(new URL("../scripts/fetch-captcha-backgrounds.config.json", import.meta.url));
	const config = JSON.parse(
		await readRepoFile("scripts/fetch-captcha-backgrounds.config.json"),
	);

	assert.deepEqual(config.sources, ["met"]);
	assert.equal(config.outputDir, "public/captcha/backgrounds");
	assert.equal(config.cacheDir, "scripts/cache/captcha");
	assert.equal(config.targetCount, 1000);
	assert.equal(config.maxPerKeyword, 300);
	assert.equal(config.minLongEdge, 1000);
	assert.ok(config.keywords.includes("landscape"));
	assert.ok(config.keywords.includes("landscape painting"));
	assert.ok(config.keywords.includes("mountain"));
	assert.ok(config.keywords.includes("forest"));
	assert.ok(config.keywords.includes("bridge"));
	assert.ok(config.keywords.includes("seascape"));
	assert.ok(config.keywords.includes("harbor"));
	assert.ok(config.keywords.includes("cityscape"));
	assert.ok(config.keywords.includes("street"));
	assert.ok(config.keywords.includes("street scene"));
	assert.ok(config.keywords.includes("architecture"));
	assert.ok(config.keywords.includes("ocean"));
	assert.ok(config.keywords.includes("island"));
	assert.ok(config.keywords.includes("ruins"));
	assert.ok(config.keywords.includes("facade"));
	assert.ok(config.keywords.includes("doorway"));
	assert.ok(config.keywords.includes("staircase"));
	assert.ok(config.keywords.includes("arcade"));
	assert.ok(config.keywords.includes("cloister"));
	assert.ok(config.keywords.includes("station"));
	assert.ok(config.keywords.includes("pavilion"));
	assert.ok(config.keywords.includes("tower"));
	assert.ok(config.keywords.includes("gate"));
	assert.ok(config.keywords.includes("nave"));
	assert.ok(config.keywords.includes("piazza"));
	assert.ok(config.keywords.includes("portico"));
	assert.ok(config.keywords.includes("colonnade"));
	assert.ok(config.keywords.includes("cathedral"));
	assert.ok(config.keywords.includes("monastery"));
	assert.ok(config.keywords.includes("church interior"));
	assert.ok(config.keywords.includes("city gate"));
	assert.ok(config.keywords.includes("alley"));
	assert.ok(config.keywords.includes("boulevard"));
	assert.ok(config.keywords.includes("terrace"));
	assert.ok(config.keywords.includes("balcony"));
});

test("captcha background fetcher produces manifest entries with stable required fields", async () => {
	const fetcherModule = await import("../scripts/fetch-captcha-backgrounds.mjs");
	const manifestEntry = fetcherModule.createManifestEntry({
		object: {
			objectID: 436121,
			title: "Sunflowers",
			objectURL: "https://www.metmuseum.org/art/collection/search/436121",
			primaryImage: "https://images.metmuseum.org/CRDImages/ep/original/DT1567.jpg",
			objectName: "Painting",
			classification: "Paintings",
			tags: [{ term: "Flowers" }],
		},
		output: {
			localPath: "/captcha/backgrounds/met/met-436121.jpg",
			width: 1800,
			height: 1200,
		},
		keyword: "flower",
		fetchedAt: "2026-04-15T00:00:00.000Z",
	});

	assert.deepEqual(manifestEntry, {
		id: "met-436121",
		source: "met",
		objectId: 436121,
		title: "Sunflowers",
		imageUrl: "https://images.metmuseum.org/CRDImages/ep/original/DT1567.jpg",
		localPath: "/captcha/backgrounds/met/met-436121.jpg",
		objectUrl: "https://www.metmuseum.org/art/collection/search/436121",
		license: "CC0",
		width: 1800,
		height: 1200,
		tags: ["flower", "painting", "paintings", "flowers"],
		fetchedAt: "2026-04-15T00:00:00.000Z",
	});
});

test("captcha background fetcher retries after an aborted timeout and then succeeds", async () => {
	const { fetchWithRetry } = await import("../scripts/fetch-captcha-backgrounds.mjs");
	let attempts = 0;

	const payload = await fetchWithRetry("https://example.com/image.jpg", {
		timeoutMs: 5,
		retries: 1,
		fetchImpl(_url, options) {
			attempts += 1;
			if (attempts === 1) {
				return new Promise((resolve, reject) => {
					options.signal.addEventListener(
						"abort",
						() => reject(new Error("aborted")),
						{ once: true },
					);
				});
			}

			return Promise.resolve({
				ok: true,
				status: 200,
				statusText: "OK",
				arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
			});
		},
		parseResponse(response) {
			return response.arrayBuffer();
		},
	});

	assert.equal(attempts, 2);
	assert.deepEqual(Array.from(new Uint8Array(payload)), [1, 2, 3]);
});

test("captcha background fetcher merges existing and new manifest entries without duplicates", async () => {
	const { mergeManifestEntries } = await import("../scripts/fetch-captcha-backgrounds.mjs");
	const mergedEntries = mergeManifestEntries([
		{
			id: "met-100",
			source: "met",
			objectId: 100,
			title: "Existing",
			imageUrl: "https://example.com/100.jpg",
			localPath: "/captcha/backgrounds/met/met-100.jpg",
			objectUrl: "https://example.com/object/100",
			license: "CC0",
			width: 1200,
			height: 900,
			tags: ["landscape"],
			fetchedAt: "2026-04-15T10:00:00.000Z",
		},
	], [
		{
			id: "met-101",
			source: "met",
			objectId: 101,
			title: "New",
			imageUrl: "https://example.com/101.jpg",
			localPath: "/captcha/backgrounds/met/met-101.jpg",
			objectUrl: "https://example.com/object/101",
			license: "CC0",
			width: 1400,
			height: 1000,
			tags: ["bridge"],
			fetchedAt: "2026-04-15T10:10:00.000Z",
		},
		{
			id: "met-100",
			source: "met",
			objectId: 100,
			title: "Existing duplicate",
			imageUrl: "https://example.com/100-duplicate.jpg",
			localPath: "/captcha/backgrounds/met/met-100.jpg",
			objectUrl: "https://example.com/object/100",
			license: "CC0",
			width: 1200,
			height: 900,
			tags: ["landscape"],
			fetchedAt: "2026-04-15T10:20:00.000Z",
		},
	]);

	assert.deepEqual(
		mergedEntries.map((entry) => entry.id),
		["met-100", "met-101"],
	);
	assert.equal(mergedEntries[0].title, "Existing");
	assert.equal(mergedEntries[1].title, "New");
});

test("captcha background fetcher accepts clearly scenic records for newly added entries", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "harbor",
			object: {
				title: "View of a Bay with a Central Tree in an Ornamental Frame",
				objectName: "Print",
				classification: "Prints",
				department: "Drawings and Prints",
				medium: "Etching",
				tags: [{ term: "Bay" }, { term: "Trees" }],
			},
		}),
		true,
	);
});

test("captcha background fetcher rejects portrait-like records even when a scenic keyword surfaced them", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "landscape",
			object: {
				title: "Bust-Length Study of a Man",
				objectName: "Painting",
				classification: "Paintings",
				department: "European Paintings",
				medium: "Oil on canvas",
				tags: [{ term: "Portraits" }, { term: "Men" }],
			},
		}),
		false,
	);
});

test("captcha background fetcher rejects vessel-centric records even when a scenic keyword surfaced them", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "river landscape",
			object: {
				title: "Vase (Old World)",
				objectName: "Vase",
				classification: "Ceramics-Vessels",
				department: "Ancient Near Eastern Art",
				medium: "Ceramic",
				tags: [{ term: "Flowers" }],
			},
		}),
		false,
	);
});

test("captcha background fetcher accepts urban street photographs that are place-centric rather than portrait-centric", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "street",
			object: {
				title: "Holden Street, North Adams, Massachusetts",
				objectName: "Photograph",
				classification: "Photographs",
				department: "Photographs",
				medium: "Gelatin silver print",
				tags: [],
			},
		}),
		true,
	);
});

test("captcha background fetcher accepts nature-heavy ocean or rock scenes for newly added entries", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "ocean",
			object: {
				title: "Peculiar shaped Rocks on Kulangsu Island, Amoy",
				objectName: "Photograph",
				classification: "Photographs",
				department: "Photographs",
				medium: "Albumen silver print",
				tags: [],
			},
		}),
		true,
	);
});

test("captcha background fetcher accepts structural architectural scenes that provide strong rotation cues", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "staircase",
			object: {
				title: "Grand Staircase with Columns",
				objectName: "Photograph",
				classification: "Photographs",
				department: "Photographs",
				medium: "Gelatin silver print",
				tags: [],
			},
		}),
		true,
	);
});

test("captcha background fetcher rejects decorative objects even when architectural words appear in the metadata", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "tower",
			object: {
				title: "Snuffbox with a Tower and Gate",
				objectName: "Snuffbox",
				classification: "Metalwork",
				department: "European Sculpture and Decorative Arts",
				medium: "Gold and enamel",
				tags: [{ term: "Buildings" }],
			},
		}),
		false,
	);
});

test("captcha background fetcher accepts cathedral or cloister interiors that still provide strong directional structure", async () => {
	const { isLikelyScenicCaptchaObject } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelyScenicCaptchaObject({
			keyword: "cathedral interior",
			object: {
				title: "Cathedral Interior with Colonnade",
				objectName: "Photograph",
				classification: "Photographs",
				department: "Photographs",
				medium: "Gelatin silver print",
				tags: [],
			},
		}),
		true,
	);
});
