export interface GatePassRoot {
	dataset: DOMStringMap & {
		gateState?: string;
	};
}

export interface GatePassOverlay {
	hidden: boolean;
	setAttribute(name: string, value: string): void;
}

export interface GatePassContent {
	inert: boolean;
}

export interface GatePassElements {
	root: GatePassRoot;
	overlay: GatePassOverlay;
	content: GatePassContent;
}

export interface CaptchaStorageLike {
	setItem(key: string, value: string): void;
}

export function markGatePassed(
	elements: GatePassElements,
	options?: {
		storageKey?: string;
		storage?: CaptchaStorageLike;
	},
): void;
