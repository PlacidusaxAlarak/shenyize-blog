export interface Point {
	x: number;
	y: number;
}

export interface PentagonShape {
	points: Point[];
	width: number;
	height: number;
	radius: number;
}

export interface Notch {
	x: number;
	y: number;
	rotation: number;
	kind: "target" | "decoy";
}

export interface ChallengeGeometry {
	shape: PentagonShape;
	sliderStartX: number;
	targetX: number;
	targetY: number;
	targetNotch: Notch;
	decoyNotch: Notch;
	pieceRotation: number;
	minNotchDistance: number;
	safeBounds: {
		minX: number;
		maxX: number;
		minY: number;
		maxY: number;
	};
	maxTravel: number;
}

export interface CaptchaState {
	currentPieceX: number;
	sliderValue: number;
	isAnimating: boolean;
	isLocked: boolean;
	status: "idle" | "loading" | "error" | "success";
}

export function createPentagonShape(radius: number): PentagonShape;

export function createChallengeGeometry(options: {
	canvasWidth: number;
	canvasHeight: number;
	pieceRadius: number;
	sliderStartX: number;
	padding: number;
	rng?: () => number;
}): ChallengeGeometry;

export function createFreshCaptchaState(options: {
	sliderStartX: number;
}): CaptchaState;

export function sliderValueToPieceX(options: {
	sliderValue: number;
	sliderStartX: number;
	maxTravel: number;
}): number;

export function evaluateAttempt(options: {
	pieceX: number;
	targetX: number;
	tolerancePx: number;
}): {
	success: boolean;
	delta: number;
};
