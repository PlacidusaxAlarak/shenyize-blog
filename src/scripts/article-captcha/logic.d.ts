export interface Point {
	x: number;
	y: number;
}

export interface RotateChallenge {
	circleCenter: Point;
	circleRadius: number;
	safeBounds: {
		minX: number;
		maxX: number;
		minY: number;
		maxY: number;
	};
	sliderMinValue: number;
	sliderMaxValue: number;
	startSliderValue: number;
	targetSliderValue: number;
	rotationSpanDeg: number;
	degreesPerSliderUnit: number;
	startRotationDeg: number;
	targetRotationDeg: number;
}

export interface CaptchaState {
	currentRotationDeg: number;
	sliderValue: number;
	startRotationDeg: number;
	startSliderValue: number;
	isAnimating: boolean;
	isLocked: boolean;
	status: "idle" | "loading" | "error" | "success";
}

export function resolveCanvasSize(options: {
	sourceWidth: number;
	sourceHeight: number;
	maxCanvasWidth: number;
}): {
	canvasWidth: number;
	canvasHeight: number;
};

export function resolveCircleRadius(options: {
	canvasWidth: number;
	canvasHeight: number;
	padding: number;
	circleRadiusRatio?: number;
}): number;

export function createRotateChallenge(options: {
	canvasWidth: number;
	canvasHeight: number;
	circleRadius: number;
	padding: number;
	sliderMinValue?: number;
	sliderMaxValue?: number;
	minTravelTurns?: number;
	maxTravelTurns?: number;
	targetSliderPaddingRatio?: number;
	rng?: () => number;
}): RotateChallenge;

export function createFreshCaptchaState(options: {
	startRotationDeg: number;
	startSliderValue: number;
}): CaptchaState;

export function sliderValueToRotation(options: {
	sliderValue: number;
	challenge: RotateChallenge;
}): number;

export function evaluateRotationAttempt(options: {
	currentRotationDeg: number;
	targetRotationDeg: number;
	toleranceDeg: number;
}): {
	success: boolean;
	deltaDeg: number;
};

export function interpolateRotationDeg(options: {
	fromDeg: number;
	toDeg: number;
	progress: number;
}): number;
