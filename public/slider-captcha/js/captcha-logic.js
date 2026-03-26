const TWO_PI = Math.PI * 2;
const SIDES = 5;
const MAX_ROTATION_DEGREES = 28;
const MIN_DECOY_ROTATION_OFFSET_DEGREES = 18;
const MAX_DECOY_ROTATION_OFFSET_DEGREES = 54;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function toRadians(degrees) {
  return (degrees * Math.PI) / 180;
}

function normalizeAngle(angle) {
  let normalized = angle;

  while (normalized > Math.PI) {
    normalized -= TWO_PI;
  }

  while (normalized <= -Math.PI) {
    normalized += TWO_PI;
  }

  return normalized;
}

function pickCoordinate(min, max, rng) {
  if (max <= min) {
    return min;
  }

  const unit = clamp(rng(), 0, 0.999999);
  return Math.round(min + (max - min) * unit);
}

function pickRange(min, max, rng) {
  const unit = clamp(rng(), 0, 0.999999);
  return min + (max - min) * unit;
}

function distanceBetween(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function isFarEnough(candidate, excludedOrigins, minDistance) {
  return excludedOrigins.every((origin) => distanceBetween(candidate, origin) >= minDistance);
}

function fallbackOrigin(safeBounds, excludedOrigins, minDistance) {
  const stepX = Math.max(12, Math.floor((safeBounds.maxX - safeBounds.minX) / 6));
  const stepY = Math.max(12, Math.floor((safeBounds.maxY - safeBounds.minY) / 5));
  let bestCandidate = { x: safeBounds.minX, y: safeBounds.minY };
  let bestDistance = -Infinity;

  for (let y = safeBounds.minY; y <= safeBounds.maxY; y += stepY) {
    for (let x = safeBounds.minX; x <= safeBounds.maxX; x += stepX) {
      const candidate = { x, y };
      const shortestDistance = excludedOrigins.length === 0
        ? Infinity
        : Math.min(...excludedOrigins.map((origin) => distanceBetween(candidate, origin)));

      if (shortestDistance > bestDistance) {
        bestDistance = shortestDistance;
        bestCandidate = candidate;
      }

      if (isFarEnough(candidate, excludedOrigins, minDistance)) {
        return candidate;
      }
    }
  }

  return bestCandidate;
}

function pickOrigin(safeBounds, rng, excludedOrigins = [], minDistance = 0) {
  for (let attempt = 0; attempt < 48; attempt += 1) {
    const candidate = {
      x: pickCoordinate(safeBounds.minX, safeBounds.maxX, rng),
      y: pickCoordinate(safeBounds.minY, safeBounds.maxY, rng)
    };

    if (isFarEnough(candidate, excludedOrigins, minDistance)) {
      return candidate;
    }
  }

  return fallbackOrigin(safeBounds, excludedOrigins, minDistance);
}

function pickPieceRotation(rng) {
  return toRadians(pickRange(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES, rng));
}

function pickDecoyRotation(pieceRotation, rng) {
  const offset = toRadians(
    pickRange(MIN_DECOY_ROTATION_OFFSET_DEGREES, MAX_DECOY_ROTATION_OFFSET_DEGREES, rng)
  );
  const signedOffset = (rng() < 0.5 ? -1 : 1) * offset;

  return normalizeAngle(pieceRotation + signedOffset);
}

export function createPentagonShape(radius) {
  const rawPoints = Array.from({ length: SIDES }, (_, index) => {
    const angle = -Math.PI / 2 + (TWO_PI * index) / SIDES;

    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius
    };
  });

  const xs = rawPoints.map((point) => point.x);
  const ys = rawPoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    points: rawPoints.map((point) => ({
      x: point.x - minX,
      y: point.y - minY
    })),
    width: maxX - minX,
    height: maxY - minY,
    radius
  };
}

export function createChallengeGeometry({
  canvasWidth,
  canvasHeight,
  pieceRadius,
  sliderStartX,
  padding,
  rng = Math.random
}) {
  const shape = createPentagonShape(pieceRadius);
  const rotationInsetX = pieceRadius - shape.width / 2;
  const rotationInsetY = pieceRadius - shape.height / 2;
  const safeBounds = {
    minX: Math.max(padding + rotationInsetX, sliderStartX + pieceRadius * 2 + 36),
    maxX: canvasWidth - padding - rotationInsetX - shape.width,
    minY: padding + rotationInsetY,
    maxY: canvasHeight - padding - rotationInsetY - shape.height
  };

  if (safeBounds.minX > safeBounds.maxX || safeBounds.minY > safeBounds.maxY) {
    throw new Error("Canvas dimensions are too small for the configured pentagon size.");
  }

  const minNotchDistance = Math.max(shape.width * 1.25, pieceRadius * 2.4);
  const targetOrigin = pickOrigin(safeBounds, rng);
  const decoyOrigin = pickOrigin(safeBounds, rng, [targetOrigin], minNotchDistance);
  const pieceRotation = pickPieceRotation(rng);
  const targetNotch = {
    x: targetOrigin.x,
    y: targetOrigin.y,
    rotation: pieceRotation,
    kind: "target"
  };
  const decoyNotch = {
    x: decoyOrigin.x,
    y: decoyOrigin.y,
    rotation: pickDecoyRotation(pieceRotation, rng),
    kind: "decoy"
  };

  return {
    shape,
    sliderStartX,
    targetX: targetNotch.x,
    targetY: targetNotch.y,
    targetNotch,
    decoyNotch,
    pieceRotation,
    minNotchDistance,
    safeBounds,
    maxTravel: canvasWidth - padding - rotationInsetX - shape.width - sliderStartX
  };
}

export function createFreshCaptchaState({ sliderStartX }) {
  return {
    currentPieceX: sliderStartX,
    sliderValue: 0,
    isAnimating: false,
    isLocked: false,
    status: "idle"
  };
}

export function sliderValueToPieceX({ sliderValue, sliderStartX, maxTravel }) {
  return sliderStartX + clamp(sliderValue, 0, maxTravel);
}

export function evaluateAttempt({ pieceX, targetX, tolerancePx }) {
  // Compare the shared bounding-box origin X for the draggable piece and the real notch.
  // The piece and real target reuse the same pentagon points and rotation, so matching the
  // translated origin X is the most stable way to compare alignment without re-deriving centers.
  const delta = Math.abs(pieceX - targetX);

  return {
    success: delta <= tolerancePx,
    delta
  };
}
