const DEFAULT_IMAGE_TIMEOUT_MS = 3000;

function loadImage(
	source,
	{ timeoutMs = DEFAULT_IMAGE_TIMEOUT_MS, imageFactory = () => new Image() } = {},
) {
	return new Promise((resolve, reject) => {
		const image = imageFactory();
		const normalizedTimeoutMs =
			Number.isFinite(timeoutMs) && timeoutMs > 0 ? Math.round(timeoutMs) : null;
		let settled = false;
		const cleanup = () => {
			image.onload = null;
			image.onerror = null;
			if (timeoutId !== null) {
				clearTimeout(timeoutId);
			}
		};
		const settle = (callback) => {
			if (settled) {
				return;
			}

			settled = true;
			cleanup();
			callback();
		};
		const timeoutId =
			normalizedTimeoutMs === null
				? null
				: setTimeout(() => {
					settle(() => reject(new Error(`Timed out loading image: ${source}`)));
				}, normalizedTimeoutMs);

		image.onload = () => settle(() => resolve(image));
		image.onerror = () => settle(() => reject(new Error(`Unable to load image: ${source}`)));
		image.src = source;
	});
}

function toRadians(degrees) {
	return (degrees * Math.PI) / 180;
}

function createCirclePath(centerX, centerY, radius) {
	const path = new Path2D();
	path.arc(centerX, centerY, radius, 0, Math.PI * 2);
	return path;
}

function getSeamColor(status) {
	if (status === "success") {
		return "rgba(29, 107, 76, 0.16)";
	}

	if (status === "error") {
		return "rgba(155, 59, 24, 0.16)";
	}

	return "rgba(20, 30, 42, 0.12)";
}

function drawRotatedPiece({
	context,
	circlePath,
	backgroundImage,
	canvas,
	centerX,
	centerY,
	rotationDeg,
}) {
	context.save();
	context.translate(centerX, centerY);
	context.rotate(toRadians(rotationDeg));
	context.translate(-centerX, -centerY);
	context.clip(circlePath);
	context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

	const highlight = context.createLinearGradient(
		centerX - 24,
		centerY - 24,
		centerX + 24,
		centerY + 24,
	);
	highlight.addColorStop(0, "rgba(255, 255, 255, 0.08)");
	highlight.addColorStop(0.55, "rgba(255, 255, 255, 0.03)");
	highlight.addColorStop(1, "rgba(28, 29, 32, 0.02)");
	context.fillStyle = highlight;
	context.fill(circlePath);
	context.restore();
}

function drawSoftSeam(context, circlePath, seamColor) {
	context.save();
	context.strokeStyle = seamColor;
	context.lineWidth = 1.25;
	context.shadowColor = "rgba(255, 255, 255, 0.12)";
	context.shadowBlur = 8;
	context.stroke(circlePath);
	context.restore();
}

export async function loadBackgroundImage(primarySource, fallbackSource, options) {
	try {
		return {
			image: await loadImage(primarySource, options),
			usedFallback: false,
		};
	} catch (primaryError) {
		if (!fallbackSource) {
			throw primaryError;
		}

		try {
			return {
				image: await loadImage(fallbackSource, options),
				usedFallback: true,
			};
		} catch (fallbackError) {
			throw new AggregateError(
				[primaryError, fallbackError],
				"Unable to load both the configured and fallback backgrounds.",
			);
		}
	}
}

export function renderCaptchaScene({
	canvas,
	context,
	backgroundImage,
	challenge,
	rotationDeg,
	status,
}) {
	const {
		circleCenter: { x: centerX, y: centerY },
		circleRadius,
	} = challenge;
	const circlePath = createCirclePath(centerX, centerY, circleRadius);

	context.clearRect(0, 0, canvas.width, canvas.height);
	context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
	drawRotatedPiece({
		context,
		circlePath,
		backgroundImage,
		canvas,
		centerX,
		centerY,
		rotationDeg,
	});
	drawSoftSeam(context, circlePath, getSeamColor(status));
}
