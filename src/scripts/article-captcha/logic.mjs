const FULL_ROTATION_DEG = 360;
const DEFAULT_TARGET_ROTATION_DEG = 0;
const MIN_CIRCLE_RADIUS = 24;
const MIN_SENSITIVITY_SCALE = 0.65;
const MAX_SENSITIVITY_SCALE = 1.35;

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

function pickSensitivityScale({ rng }) {
	const unit = clampUnitInterval(rng());
	return MIN_SENSITIVITY_SCALE + (MAX_SENSITIVITY_SCALE - MIN_SENSITIVITY_SCALE) * unit;
}

function pickRotationDirection({ rng }) {
	return rng() < 0.5 ? -1 : 1;
}

function pickSliderStartValue({ rng, sliderMinValue, sliderMaxValue }) {
	return rng() < 0.5 ? sliderMinValue : sliderMaxValue;
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

export function resolveVisibleCanvasLimit({
	viewportWidth,
	viewportHeight,
	overlayPadding,
	cardPaddingX,
	cardPaddingY,
	headerHeight,
	contentGap,
	framePaddingX,
	framePaddingY,
	maxCanvasWidth,
}) {
	if (
		viewportWidth <= 0 ||
		viewportHeight <= 0 ||
		overlayPadding < 0 ||
		cardPaddingX < 0 ||
		cardPaddingY < 0 ||
		headerHeight < 0 ||
		contentGap < 0 ||
		framePaddingX < 0 ||
		framePaddingY < 0 ||
		maxCanvasWidth <= 0
	) {
		throw new Error("Visible canvas limit inputs must be non-negative numbers.");
	}

	const availableWidth = viewportWidth - overlayPadding * 2 - cardPaddingX - framePaddingX;
	const availableHeight =
		viewportHeight -
		overlayPadding * 2 -
		cardPaddingY -
		headerHeight -
		contentGap -
		framePaddingY;

	if (availableWidth <= 0 || availableHeight <= 0) {
		throw new Error("Viewport is too small to fit the captcha image.");
	}

	return Math.max(1, Math.floor(Math.min(maxCanvasWidth, availableWidth, availableHeight)));
}

function resolveWeightedCandidate(candidates, rng) {
	const weightedCandidates = candidates
		.map((candidate) => {
			const width = Math.max(candidate.maxLeft - candidate.minLeft, 0) + 1;
			const height = Math.max(candidate.maxTop - candidate.minTop, 0) + 1;

			return {
				...candidate,
				weight: width * height,
			};
		})
		.filter((candidate) => candidate.weight > 0);

	const totalWeight = weightedCandidates.reduce((sum, candidate) => sum + candidate.weight, 0);
	let threshold = clampUnitInterval(rng()) * totalWeight;

	for (const candidate of weightedCandidates) {
		threshold -= candidate.weight;
		if (threshold <= 0) {
			return candidate;
		}
	}

	return weightedCandidates[weightedCandidates.length - 1];
}

function pickCoordinate({ min, max, rng }) {
	if (min > max) {
		throw new Error("Coordinate range must be ordered.");
	}

	if (min === max) {
		return Math.round(min);
	}

	return Math.round(min + (max - min) * clampUnitInterval(rng()));
}

function createCandidateArea({ minLeft, maxLeft, minTop, maxTop }) {
	if (minLeft > maxLeft || minTop > maxTop) {
		return null;
	}

	return { minLeft, maxLeft, minTop, maxTop };
}

export function resolveFloatingPanelPosition({
	viewportWidth,
	viewportHeight,
	panelWidth,
	panelHeight,
	blockedRect,
	padding = 18,
	gap = 18,
	rng = Math.random,
}) {
	if (viewportWidth <= 0 || viewportHeight <= 0 || panelWidth <= 0 || panelHeight <= 0) {
		throw new Error("Viewport and panel dimensions must be positive numbers.");
	}

	const safeMinLeft = padding;
	const safeMinTop = padding;
	const safeMaxLeft = viewportWidth - padding - panelWidth;
	const safeMaxTop = viewportHeight - padding - panelHeight;

	if (safeMinLeft > safeMaxLeft || safeMinTop > safeMaxTop) {
		throw new Error("Viewport is too small to fit the floating panel.");
	}

	if (!blockedRect) {
		return {
			left: pickCoordinate({ min: safeMinLeft, max: safeMaxLeft, rng }),
			top: pickCoordinate({ min: safeMinTop, max: safeMaxTop, rng }),
		};
	}

	const expandedBlockedRect = {
		left: blockedRect.left - gap,
		top: blockedRect.top - gap,
		right: blockedRect.right + gap,
		bottom: blockedRect.bottom + gap,
	};
	const candidates = [
		createCandidateArea({
			minLeft: safeMinLeft,
			maxLeft: safeMaxLeft,
			minTop: safeMinTop,
			maxTop: Math.min(safeMaxTop, expandedBlockedRect.top - panelHeight),
		}),
		createCandidateArea({
			minLeft: safeMinLeft,
			maxLeft: safeMaxLeft,
			minTop: Math.max(safeMinTop, expandedBlockedRect.bottom),
			maxTop: safeMaxTop,
		}),
		createCandidateArea({
			minLeft: safeMinLeft,
			maxLeft: Math.min(safeMaxLeft, expandedBlockedRect.left - panelWidth),
			minTop: safeMinTop,
			maxTop: safeMaxTop,
		}),
		createCandidateArea({
			minLeft: Math.max(safeMinLeft, expandedBlockedRect.right),
			maxLeft: safeMaxLeft,
			minTop: safeMinTop,
			maxTop: safeMaxTop,
		}),
	].filter(Boolean);

	if (candidates.length === 0) {
		if (gap > 0) {
			return resolveFloatingPanelPosition({
				viewportWidth,
				viewportHeight,
				panelWidth,
				panelHeight,
				blockedRect,
				padding,
				gap: 0,
				rng,
			});
		}

		throw new Error("Unable to place the floating panel without overlapping the blocked rect.");
	}

	const selectedCandidate = resolveWeightedCandidate(candidates, rng);

	return {
		left: pickCoordinate({
			min: selectedCandidate.minLeft,
			max: selectedCandidate.maxLeft,
			rng,
		}),
		top: pickCoordinate({
			min: selectedCandidate.minTop,
			max: selectedCandidate.maxTop,
			rng,
		}),
	};
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
	const startSliderValue = pickSliderStartValue({
		rng,
		sliderMinValue,
		sliderMaxValue,
	});
	const rotationSpanDeg = pickTravelSpanDeg({
		rng,
		minTravelTurns,
		maxTravelTurns,
	});
	const sensitivityScale = pickSensitivityScale({ rng });
	const rotationDirection = pickRotationDirection({ rng });
	const degreesPerSliderUnit = rotationSpanDeg / (sliderMaxValue - sliderMinValue);
	const startRotationDeg = normalizeRotationDeg(
		DEFAULT_TARGET_ROTATION_DEG +
			(startSliderValue - targetSliderValue) *
				degreesPerSliderUnit *
				sensitivityScale *
				rotationDirection,
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
		sensitivityScale,
		rotationDirection,
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
			(nextSliderValue - challenge.targetSliderValue) *
				challenge.degreesPerSliderUnit *
				(challenge.sensitivityScale ?? 1) *
				(challenge.rotationDirection ?? 1),
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
