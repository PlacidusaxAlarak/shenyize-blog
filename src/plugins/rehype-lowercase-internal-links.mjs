const schemePattern = /^[a-zA-Z][\w+.-]*:/;

function shouldIgnoreHref(href) {
	const lowered = href.toLowerCase();
	return (
		lowered.startsWith("#") ||
		lowered.startsWith("//") ||
		lowered.startsWith("mailto:") ||
		lowered.startsWith("tel:") ||
		lowered.startsWith("javascript:")
	);
}

function normalizeInternalHref(href) {
	const raw = href.trim();
	if (!raw || shouldIgnoreHref(raw)) return href;

	const isRelative = !raw.startsWith("/") && !schemePattern.test(raw);
	let url;
	try {
		url = new URL(raw, "https://example.com");
	} catch {
		return href;
	}
	const lowerPath = url.pathname.toLowerCase();

	if (!lowerPath.startsWith("/posts/") && !lowerPath.startsWith("/solutions/")) {
		return href;
	}

	const rebuilt = `${lowerPath}${url.search}${url.hash}`;
	return isRelative ? rebuilt.replace(/^\//, "") : rebuilt;
}

function visit(node) {
	if (!node || typeof node !== "object") return;
	if (
		node.type === "element" &&
		node.tagName === "a" &&
		node.properties &&
		typeof node.properties.href === "string"
	) {
		const nextHref = normalizeInternalHref(node.properties.href);
		if (nextHref !== node.properties.href) {
			node.properties.href = nextHref;
		}
	}

	if (Array.isArray(node.children)) {
		for (const child of node.children) {
			visit(child);
		}
	}
}

export function rehypeLowercaseInternalLinks() {
	return (tree) => {
		visit(tree);
	};
}
