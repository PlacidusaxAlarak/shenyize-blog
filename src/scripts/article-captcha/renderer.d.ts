import type { CaptchaState, RotateChallenge } from "./logic.mjs";

export interface LoadBackgroundImageOptions {
	timeoutMs?: number;
	imageFactory?: () => HTMLImageElement;
}

export function loadBackgroundImage(
	primarySource: string,
	options?: LoadBackgroundImageOptions,
): Promise<HTMLImageElement>;

export function loadBackgroundImageFromSources(
	sources: string[],
	options?: LoadBackgroundImageOptions,
): Promise<{
	image: HTMLImageElement;
	source: string;
}>;

export function renderCaptchaScene(options: {
	canvas: HTMLCanvasElement;
	context: CanvasRenderingContext2D;
	backgroundImage: CanvasImageSource;
	challenge: RotateChallenge;
	rotationDeg: number;
	status: CaptchaState["status"];
}): void;
