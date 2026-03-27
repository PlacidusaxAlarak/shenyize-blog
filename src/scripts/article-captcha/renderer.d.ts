import type { CaptchaState, ChallengeGeometry } from "./logic.mjs";

export interface LoadBackgroundImageOptions {
	timeoutMs?: number;
	imageFactory?: () => HTMLImageElement;
}

export function loadBackgroundImage(
	primarySource: string,
	fallbackSource?: string,
	options?: LoadBackgroundImageOptions,
): Promise<{
	image: HTMLImageElement;
	usedFallback: boolean;
}>;

export function renderCaptchaScene(options: {
	canvas: HTMLCanvasElement;
	context: CanvasRenderingContext2D;
	backgroundImage: CanvasImageSource;
	geometry: ChallengeGeometry;
	pieceX: number;
	status: CaptchaState["status"];
}): void;
