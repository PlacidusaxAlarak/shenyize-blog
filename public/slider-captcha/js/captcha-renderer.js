function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();

    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load image: ${source}`));
    image.src = source;
  });
}

function rotatePoint(point, centerX, centerY, rotation) {
  const dx = point.x - centerX;
  const dy = point.y - centerY;
  const cosine = Math.cos(rotation);
  const sine = Math.sin(rotation);

  return {
    x: centerX + dx * cosine - dy * sine,
    y: centerY + dx * sine + dy * cosine
  };
}

function buildRotatedPentagonPath({ points, width, height, originX, originY, rotation }) {
  const centerX = originX + width / 2;
  const centerY = originY + height / 2;
  const [firstPoint, ...otherPoints] = points;
  const path = new Path2D();

  // The pentagon points are normalized into a tight local box once, then rotated around
  // that box center after translation. Reusing the same normalized points for the real
  // notch, the decoy notch, and the floating piece keeps every silhouette consistent.
  const rotatedFirst = rotatePoint(
    { x: originX + firstPoint.x, y: originY + firstPoint.y },
    centerX,
    centerY,
    rotation
  );
  path.moveTo(rotatedFirst.x, rotatedFirst.y);

  for (const point of otherPoints) {
    const rotatedPoint = rotatePoint(
      { x: originX + point.x, y: originY + point.y },
      centerX,
      centerY,
      rotation
    );

    path.lineTo(rotatedPoint.x, rotatedPoint.y);
  }

  path.closePath();
  return path;
}

function drawNotch(context, path, kind) {
  const isDecoy = kind === "decoy";

  context.save();
  context.fillStyle = isDecoy ? "rgba(91, 58, 33, 0.18)" : "rgba(20, 30, 42, 0.26)";
  context.shadowColor = isDecoy ? "rgba(148, 92, 48, 0.22)" : "rgba(20, 30, 42, 0.34)";
  context.shadowBlur = 18;
  context.fill(path);
  context.restore();

  context.save();
  context.strokeStyle = isDecoy ? "rgba(255, 223, 187, 0.42)" : "rgba(255, 255, 255, 0.56)";
  context.lineWidth = 1.4;
  context.stroke(path);
  context.restore();
}

export async function loadBackgroundImage(primarySource, fallbackSource) {
  try {
    return {
      image: await loadImage(primarySource),
      usedFallback: false
    };
  } catch (primaryError) {
    if (!fallbackSource) {
      throw primaryError;
    }

    try {
      return {
        image: await loadImage(fallbackSource),
        usedFallback: true
      };
    } catch (fallbackError) {
      throw new AggregateError([primaryError, fallbackError], "Unable to load both the configured and fallback backgrounds.");
    }
  }
}

export function renderCaptchaScene({ canvas, context, backgroundImage, geometry, pieceX, status }) {
  const { shape, targetNotch, decoyNotch, pieceRotation } = geometry;
  const targetPath = buildRotatedPentagonPath({
    points: shape.points,
    width: shape.width,
    height: shape.height,
    originX: targetNotch.x,
    originY: targetNotch.y,
    rotation: targetNotch.rotation
  });
  const decoyPath = buildRotatedPentagonPath({
    points: shape.points,
    width: shape.width,
    height: shape.height,
    originX: decoyNotch.x,
    originY: decoyNotch.y,
    rotation: decoyNotch.rotation
  });
  const piecePath = buildRotatedPentagonPath({
    points: shape.points,
    width: shape.width,
    height: shape.height,
    originX: pieceX,
    originY: targetNotch.y,
    rotation: pieceRotation
  });
  const pieceAccent =
    status === "success"
      ? "rgba(29, 107, 76, 0.4)"
      : status === "error"
        ? "rgba(155, 59, 24, 0.38)"
        : "rgba(28, 29, 32, 0.26)";

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

  drawNotch(context, decoyPath, "decoy");
  drawNotch(context, targetPath, "target");

  context.save();
  context.shadowColor = "rgba(20, 30, 42, 0.28)";
  context.shadowBlur = 18;
  context.shadowOffsetY = 10;
  context.fillStyle = pieceAccent;
  context.fill(piecePath);
  context.restore();

  context.save();
  context.clip(piecePath);
  context.drawImage(backgroundImage, pieceX - targetNotch.x, 0, canvas.width, canvas.height);

  const highlight = context.createLinearGradient(pieceX, targetNotch.y, pieceX + shape.width, targetNotch.y + shape.height);
  highlight.addColorStop(0, "rgba(255, 255, 255, 0.28)");
  highlight.addColorStop(0.55, "rgba(255, 255, 255, 0.08)");
  highlight.addColorStop(1, "rgba(28, 29, 32, 0.1)");
  context.fillStyle = highlight;
  context.fill(piecePath);
  context.restore();

  context.save();
  context.strokeStyle = "rgba(255, 255, 255, 0.72)";
  context.lineWidth = 1.8;
  context.stroke(piecePath);
  context.restore();
}
