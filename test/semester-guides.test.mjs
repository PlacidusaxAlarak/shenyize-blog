import assert from "node:assert/strict";
import test from "node:test";
import { getSemesterGuides } from "../src/utils/semester-guides.mjs";

test("semester index includes published guides in academic order", () => {
	const posts = [
		{ slug: "suda-se-guide/大二下", data: { draft: false } },
		{ slug: "suda-se-guide/大三上", data: {} },
		{ slug: "suda-se-guide/大四下", data: {} },
		{ slug: "suda-se-guide/大一上", data: {} },
		{ slug: "suda-se-guide/大四上", data: {} },
		{ slug: "suda-se-guide/大二上", data: {} },
		{ slug: "suda-se-guide/大一下", data: {} },
		{ slug: "suda-se-guide/大三下", data: { draft: true } },
		{ slug: "suda-se-guide/大四上/notes", data: {} },
		{ slug: "guild", data: {} },
	];

	assert.deepEqual(
		getSemesterGuides(posts),
		["大一上", "大一下", "大二上", "大二下", "大三上", "大四上", "大四下"].map((semester) => ({
			slug: `suda-se-guide/${semester}`,
			semester,
		})),
	);
});
