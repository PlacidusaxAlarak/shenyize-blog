const FULL_ROTATION_DEG = 360;
const DEFAULT_TARGET_ROTATION_DEG = 0;
const MIN_CIRCLE_RADIUS = 24;

function clamp(value, min, max) {
	return Math.min(Math.max(value, min), max);
}

function clampUnitInterval(value) {
	return clamp(value, 0, 0.999999);
}

function normalizeRotationDeg(value) {
	const normalized = value % FULL_ROTATION_DEG;

	return normalized < 0 ? normalized + FULL_ROTATION_DEG : normalized;
}

function getRotationDeltaDeg(fromDeg, toDeg) {
	const from = normalizeRotationDeg(fromDeg);
	const to = normalizeRotationDeg(toDeg);
	let delta = to - from;

	if (delta > 180) {
		delta -= FULL_ROTATION_DEG;
	}

	if (delta <= -180) {
		delta += FULL_ROTATION_DEG;
	}

	return delta;
}

function getShortestDistanceDeg(firstDeg, secondDeg) {
	return Math.abs(getRotationDeltaDeg(firstDeg, secondDeg));
}

function pickTravelSpanDeg({ rng, minTravelTurns, maxTravelTurns }) {
	if (minTravelTurns <= 0 || maxTravelTurns <= 0 || minTravelTurns > maxTravelTurns) {
		throw new Error("Travel turn range must be positive and ordered.");
	}

	const unit = clampUnitInterval(rng());
	const turns = minTravelTurns + (maxTravelTurns - minTravelTurns) * unit;

	return turns * FULL_ROTATION_DEG;
}

function pickTargetSliderValue({ rng, sliderMinValue, sliderMaxValue, targetSliderPaddingRatio }) {
	const sliderSpan = sliderMaxValue - sliderMinValue;
	const padding = sliderSpan * targetSliderPaddingRatio;
	const minTarget = sliderMinValue + padding;
	const maxTarget = sliderMaxValue - padding;

	if (minTarget > maxTarget) {
		return Math.round(sliderMinValue + sliderSpan / 2);
	}

	const unit = clampUnitInterval(rng());
	return Math.round(minTarget + (maxTarget - minTarget) * unit);
}

export function resolveCanvasSize({ sourceWidth, sourceHeight, maxCanvasWidth }) {
	if (sourceWidth <= 0 || sourceHeight <= 0 || maxCanvasWidth <= 0) {
		throw new Error("Canvas size inputs must be positive numbers.");
	}

	const scale = Math.min(1, maxCanvasWidth / sourceWidth);

	return {
		canvasWidth: Math.round(sourceWidth * scale),
		canvasHeight: Math.round(sourceHeight * scale),
	};
}

export function resolveCircleRadius({
	canvasWidth,
	canvasHeight,
	padding,
	circleRadiusRatio = 0.18,
}) {
	const desiredRadius = Math.round(Math.min(canvasWidth, canvasHeight) * circleRadiusRatio);
	const maxAllowedRadius = Math.min(
		Math.floor(canvasWidth / 2) - padding,
		Math.floor(canvasHeight / 2) - padding,
	);

	if (maxAllowedRadius < MIN_CIRCLE_RADIUS) {
		throw new Error("Canvas dimensions are too small to place the centered circle.");
	}

	return clamp(desiredRadius, MIN_CIRCLE_RADIUS, maxAllowedRadius);
}

export function createRotateChallenge({
	canvasWidth,
	canvasHeight,
	circleRadius,
	padding,
	sliderMinValue = 0,
	sliderMaxValue = 100,
	minTravelTurns = 0.5,
	maxTravelTurns = 0.95,
	targetSliderPaddingRatio = 0.18,
	rng = Math.random,
}) {
	const safeBounds = {
		minX: padding + circleRadius,
		maxX: canvasWidth - padding - circleRadius,
		minY: padding + circleRadius,
		maxY: canvasHeight - padding - circleRadius,
	};
	const circleCenter = {
		x: Math.round(canvasWidth / 2),
		y: Math.round(canvasHeight / 2),
	};

	if (safeBounds.minX > safeBounds.maxX || safeBounds.minY > safeBounds.maxY) {
		throw new Error("Canvas dimensions are too small for the configured circle size.");
	}

	if (
		circleCenter.x < safeBounds.minX ||
		circleCenter.x > safeBounds.maxX ||
		circleCenter.y < safeBounds.minY ||
		circleCenter.y > safeBounds.maxY
	) {
		throw new Error(
			"Canvas dimensions are too small to keep the circle centered with the configured padding.",
		);
	}

	if (sliderMaxValue <= sliderMinValue) {
		throw new Error("Slider value range must be ordered.");
	}

	const targetSliderValue = pickTargetSliderValue({
		rng,
		sliderMinValue,
		sliderMaxValue,
		targetSliderPaddingRatio,
	});
	const startSliderValue = sliderMinValue;
	const rotationSpanDeg = pickTravelSpanDeg({
		rng,
		minTravelTurns,
		maxTravelTurns,
	});
	const degreesPerSliderUnit = rotationSpanDeg / (sliderMaxValue - sliderMinValue);
	const startRotationDeg = normalizeRotationDeg(
		DEFAULT_TARGET_ROTATION_DEG + (startSliderValue - targetSliderValue) * degreesPerSliderUnit,
	);

	return {
		circleCenter,
		circleRadius,
		safeBounds,
		sliderMinValue,
		sliderMaxValue,
		startSliderValue,
		targetSliderValue,
		rotationSpanDeg,
		degreesPerSliderUnit,
		startRotationDeg,
		targetRotationDeg: DEFAULT_TARGET_ROTATION_DEG,
	};
}

export function createFreshCaptchaState({ startRotationDeg, startSliderValue }) {
	return {
		currentRotationDeg: startRotationDeg,
		sliderValue: startSliderValue,
		startRotationDeg,
		startSliderValue,
		isAnimating: false,
		isLocked: false,
		status: "idle",
	};
}

export function sliderValueToRotation({ sliderValue, challenge }) {
	const nextSliderValue = clamp(
		Number(sliderValue),
		challenge.sliderMinValue,
		challenge.sliderMaxValue,
	);

	return normalizeRotationDeg(
		challenge.targetRotationDeg +
			(nextSliderValue - challenge.targetSliderValue) * challenge.degreesPerSliderUnit,
	);
}

export function evaluateRotationAttempt({
	currentRotationDeg,
	targetRotationDeg,
	toleranceDeg,
}) {
	const deltaDeg = getShortestDistanceDeg(currentRotationDeg, targetRotationDeg);

	return {
		success: deltaDeg <= toleranceDeg,
		deltaDeg,
	};
}

export function interpolateRotationDeg({ fromDeg, toDeg, progress }) {
	const easedProgress = clamp(progress, 0, 1);
	return normalizeRotationDeg(fromDeg + getRotationDeltaDeg(fromDeg, toDeg) * easedProgress);
}
