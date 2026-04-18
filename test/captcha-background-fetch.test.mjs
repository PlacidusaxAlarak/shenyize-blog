import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repoRoot = new URL("../", import.meta.url);
const cmaSampleModule = () => import("../scripts/fetch-cma-sample-backgrounds.mjs");
const openImagesSampleModule = () => import("../scripts/fetch-openimages-sample-backgrounds.mjs");

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

test("captcha background fetcher rejects sparse album or scroll compositions that leave most of the frame blank", async () => {
	const { isLikelySparseCaptchaPresentation } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelySparseCaptchaPresentation({
			object: {
				objectName: "Folding fan mounted as an album leaf",
				classification: "Paintings",
			},
			metrics: {
				contentRatio: 0.36,
				bboxAreaRatio: 0.95,
			},
		}),
		true,
	);
});

test("captcha background fetcher keeps ordinary scenic prints when the image content fills the frame", async () => {
	const { isLikelySparseCaptchaPresentation } = await import("../scripts/fetch-captcha-backgrounds.mjs");

	assert.equal(
		isLikelySparseCaptchaPresentation({
			object: {
				objectName: "Print",
				classification: "Prints",
			},
			metrics: {
				contentRatio: 0.36,
				bboxAreaRatio: 0.6,
			},
		}),
		false,
	);
});

test("cma sample fetch defines the scenic keyword shortlist for the five-image trial", async () => {
	const { cmaSampleKeywords } = await cmaSampleModule();

	assert.deepEqual(cmaSampleKeywords, [
		"landscape",
		"architecture",
		"bridge",
		"street",
		"garden",
	]);
});

test("cma sample fetcher creates manifest entries using the shared background contract", async () => {
	const { createCmaManifestEntry } = await cmaSampleModule();

	const manifestEntry = createCmaManifestEntry({
		artwork: {
			id: 12345,
			title: "Bridge over Water",
			url: "https://www.clevelandart.org/art/12345",
			type: "Print",
			department: "Japanese Art",
			images: {
				print: {
					url: "https://openaccess-cdn.clevelandart.org/12345/12345_print.jpg",
				},
			},
		},
		output: {
			localPath: "/captcha/backgrounds/cma/cma-12345.jpg",
			width: 2285,
			height: 3400,
		},
		keyword: "bridge",
		fetchedAt: "2026-04-16T00:00:00.000Z",
	});

	assert.deepEqual(manifestEntry, {
		id: "cma-12345",
		source: "cma",
		objectId: 12345,
		title: "Bridge over Water",
		imageUrl: "https://openaccess-cdn.clevelandart.org/12345/12345_print.jpg",
		localPath: "/captcha/backgrounds/cma/cma-12345.jpg",
		objectUrl: "https://www.clevelandart.org/art/12345",
		license: "CC0",
		width: 2285,
		height: 3400,
		tags: ["bridge", "print", "japanese art"],
		fetchedAt: "2026-04-16T00:00:00.000Z",
	});
});

test("cma sample fetcher accepts scenic architectural records and rejects decorative object records", async () => {
	const { isLikelyCmaSampleArtwork } = await cmaSampleModule();

	assert.equal(
		isLikelyCmaSampleArtwork({
			artwork: {
				share_license_status: "CC0",
				title: "Bridge in a Garden",
				type: "Print",
				department: "Japanese Art",
				collection: "Japanese Art",
				technique: "color woodblock print",
				images: {
					print: {
						url: "https://openaccess-cdn.clevelandart.org/12345/12345_print.jpg",
						width: "2285",
						height: "3400",
					},
				},
			},
			keyword: "bridge",
		}),
		true,
	);

	assert.equal(
		isLikelyCmaSampleArtwork({
			artwork: {
				share_license_status: "CC0",
				title: "Wine Ewer",
				type: "Silver",
				department: "Decorative Art and Design",
				collection: "Decorative Arts",
				technique: "silver gilt",
				images: {
					print: {
						url: "https://openaccess-cdn.clevelandart.org/1943.181/1943.181_print.jpg",
						width: "2627",
						height: "3400",
					},
				},
			},
			keyword: "bridge",
		}),
		false,
	);
});

test("cma fetch target counts total cma entries rather than adding the full limit each run", async () => {
	const { resolveCmaRemainingSlots } = await cmaSampleModule();

	assert.equal(
		resolveCmaRemainingSlots({
			manifestEntries: [
				{ id: "met-1", source: "met" },
				{ id: "cma-1", source: "cma" },
				{ id: "cma-2", source: "cma" },
				{ id: "cma-3", source: "cma" },
				{ id: "cma-4", source: "cma" },
				{ id: "cma-5", source: "cma" },
			],
			targetCount: 50,
		}),
		45,
	);

	assert.equal(
		resolveCmaRemainingSlots({
			manifestEntries: [
				{ id: "cma-1", source: "cma" },
				{ id: "cma-2", source: "cma" },
				{ id: "cma-3", source: "cma" },
			],
			targetCount: 2,
		}),
		0,
	);
});

test("cma fetch can switch to photograph-priority search keywords for supplementation", async () => {
	const { resolveCmaSearchKeywords } = await cmaSampleModule();

	assert.deepEqual(resolveCmaSearchKeywords({ preferPhotographs: true }), [
		"photograph landscape",
		"architectural photograph",
		"bridge photograph",
		"street photograph",
		"garden photograph",
		"architecture photograph",
		"city photograph",
	]);
});

test("cma candidate ranking prefers scenic photographs over paintings when photo mode is enabled", async () => {
	const { scoreCmaCandidateForSort } = await cmaSampleModule();

	const photographScore = scoreCmaCandidateForSort({
		artwork: {
			title: "Bridge of Shops, Srinagar, Kashmir",
			type: "Photograph",
			department: "Photography",
			collection: "Photography",
			technique: "albumen silver print",
		},
		keyword: "bridge photograph",
		preferPhotographs: true,
	});
	const paintingScore = scoreCmaCandidateForSort({
		artwork: {
			title: "Landscape with Fishermen",
			type: "Painting",
			department: "Korean Art",
			collection: "Asian - Hanging Scroll",
			technique: "ink and color on silk",
		},
		keyword: "landscape",
		preferPhotographs: true,
	});

	assert.ok(photographScore > paintingScore);
});

test("cma detail collection skips timed-out artwork records instead of aborting the whole run", async () => {
	const { collectCmaArtworkDetails } = await cmaSampleModule();

	const detailed = await collectCmaArtworkDetails({
		candidates: [
			{ id: 1, keyword: "bridge photograph" },
			{ id: 2, keyword: "street photograph" },
		],
		loadArtwork(candidate) {
			if (candidate.id === 1) {
				throw new Error("timeout");
			}

			return Promise.resolve({
				id: 2,
				share_license_status: "CC0",
				title: "Street Advertising",
				type: "Photograph",
				department: "Photography",
				collection: "Photography",
				technique: "gelatin silver print",
				images: {
					print: {
						url: "https://openaccess-cdn.clevelandart.org/2/2_print.jpg",
						width: "3400",
						height: "2600",
					},
				},
			});
		},
		preferPhotographs: true,
	});

	assert.deepEqual(
		detailed.map((entry) => entry.artwork.id),
		[2],
	);
});

test("open images sample fetch keeps curated seeds and expands toward a 200 image target", async () => {
	const {
		openImagesPinnedSampleTargets,
		openImagesSampleTargetCount,
		openImagesSampleThemeRules,
	} = await openImagesSampleModule();

	assert.equal(openImagesPinnedSampleTargets.length, 20);
	assert.equal(openImagesSampleTargetCount, 200);
	assert.deepEqual(
		openImagesPinnedSampleTargets.slice(0, 5).map((target) => target.imageId),
		[
			"00794645d77184eb",
			"0a3f577a327ca7cc",
			"115ef722923602a8",
			"0a556c8163b58fae",
			"0b8ba050b1d83bb7",
		],
	);
	assert.ok(
		openImagesPinnedSampleTargets.some((target) => target.subject === "forest-waterfall"),
	);
	assert.ok(
		openImagesPinnedSampleTargets.some((target) => target.subject === "spiral-stairs"),
	);
	assert.ok(
		openImagesSampleThemeRules.some((rule) => rule.theme === "architecture"),
	);
	assert.ok(
		openImagesSampleThemeRules.some((rule) => rule.theme === "waterfall"),
	);
	assert.ok(
		openImagesSampleThemeRules.some((rule) => rule.theme === "bathroom"),
	);
});

test("open images target expansion preserves pinned samples and fills remaining slots round-robin", async () => {
	const { selectOpenImagesSampleTargets } = await openImagesSampleModule();

	const expanded = selectOpenImagesSampleTargets({
		pinnedTargets: [
			{
				imageId: "seed-1",
				slug: "seed-1.jpg",
				subject: "seed-one",
				labels: ["building"],
			},
		],
		candidateTargetsByTheme: new Map([
			[
				"architecture",
				[
					{
						imageId: "a-1",
						slug: "a-1.jpg",
						subject: "architecture",
						labels: ["building"],
					},
					{
						imageId: "shared",
						slug: "shared.jpg",
						subject: "architecture",
						labels: ["building"],
					},
				],
			],
			[
				"waterfall",
				[
					{
						imageId: "w-1",
						slug: "w-1.jpg",
						subject: "waterfall",
						labels: ["waterfall"],
					},
					{
						imageId: "shared",
						slug: "shared.jpg",
						subject: "waterfall",
						labels: ["waterfall"],
					},
					{
						imageId: "w-2",
						slug: "w-2.jpg",
						subject: "waterfall",
						labels: ["waterfall"],
					},
				],
			],
		]),
		targetCount: 4,
	});

	assert.deepEqual(
		expanded.map((target) => target.imageId),
		["seed-1", "a-1", "w-1", "shared"],
	);
});

test("open images sample metadata entries preserve attribution and license fields", async () => {
	const { createOpenImagesSampleMetadataEntry } = await openImagesSampleModule();

	const entry = createOpenImagesSampleMetadataEntry({
		target: {
			imageId: "00794645d77184eb",
			slug: "openimages-garage-door.jpg",
			subject: "garage-door",
			labels: ["building", "door", "window"],
		},
		record: {
			Subset: "validation",
			ImageID: "00794645d77184eb",
			OriginalURL: "https://c4.staticflickr.com/4/3932/15256561208_17c0f4c46c_o.jpg",
			OriginalLandingURL: "https://www.flickr.com/photos/stevenpisano/15256561208",
			License: "https://creativecommons.org/licenses/by/2.0/",
			AuthorProfileURL: "https://www.flickr.com/people/stevenpisano/",
			Author: "Steven Pisano",
			Title: "Brooklyn Street Scenes",
			Thumbnail300KURL: "https://c1.staticflickr.com/4/3932/15256561208_881fdb1641_z.jpg",
			Rotation: "0.0",
		},
		output: {
			localPath: "/openimages-sample/openimages-garage-door.jpg",
			width: 2048,
			height: 1536,
		},
		fetchedAt: "2026-04-17T00:00:00.000Z",
	});

	assert.deepEqual(entry, {
		id: "openimages-00794645d77184eb",
		source: "openimages",
		imageId: "00794645d77184eb",
		subset: "validation",
		title: "Brooklyn Street Scenes",
		subject: "garage-door",
		labels: ["building", "door", "window"],
		imageUrl: "https://c4.staticflickr.com/4/3932/15256561208_17c0f4c46c_o.jpg",
		thumbnailUrl: "https://c1.staticflickr.com/4/3932/15256561208_881fdb1641_z.jpg",
		localPath: "/openimages-sample/openimages-garage-door.jpg",
		landingUrl: "https://www.flickr.com/photos/stevenpisano/15256561208",
		license: "https://creativecommons.org/licenses/by/2.0/",
		licenseFamily: "CC-BY",
		author: "Steven Pisano",
		authorProfileUrl: "https://www.flickr.com/people/stevenpisano/",
		width: 2048,
		height: 1536,
		rotation: 0,
		fetchedAt: "2026-04-17T00:00:00.000Z",
	});
});
