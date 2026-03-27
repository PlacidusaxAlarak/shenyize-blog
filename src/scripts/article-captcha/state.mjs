const STORAGE_VALUE = "passed";

function setGatePassedState({ root, overlay, content }) {
	root.dataset.gateState = "passed";
	overlay.hidden = true;
	content.inert = false;
	overlay.setAttribute("aria-hidden", "true");
}

export function markGatePassed(elements, { storageKey, storage } = {}) {
	setGatePassedState(elements);

	if (!storageKey || !storage?.setItem) {
		return;
	}

	try {
		storage.setItem(storageKey, STORAGE_VALUE);
	} catch {
		// Ignore storage failures so the gate can still unlock for the current view.
	}
}
