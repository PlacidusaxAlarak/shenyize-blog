import assert from "node:assert/strict";
import test from "node:test";

const stateModule = () => import("../src/scripts/article-captcha/state.mjs");

function createOverlayElement() {
	return {
		hidden: false,
		attributes: new Map(),
		setAttribute(name, value) {
			this.attributes.set(name, String(value));
		},
		getAttribute(name) {
			return this.attributes.get(name) ?? null;
		},
	};
}

test("article captcha marks the current gate as passed and remembers the current session", async () => {
	const { markGatePassed } = await stateModule();
	const root = {
		dataset: {
			gateState: "locked",
		},
	};
	const overlay = createOverlayElement();
	const content = {
		inert: true,
	};
	const storageEntries = new Map();
	const storage = {
		setItem(key, value) {
			storageEntries.set(key, String(value));
		},
	};

	markGatePassed(
		{ root, overlay, content },
		{ storageKey: "site-captcha:passed", storage },
	);

	assert.equal(root.dataset.gateState, "passed");
	assert.equal(overlay.hidden, true);
	assert.equal(overlay.getAttribute("aria-hidden"), "true");
	assert.equal(content.inert, false);
	assert.equal(storageEntries.get("site-captcha:passed"), "passed");
});
