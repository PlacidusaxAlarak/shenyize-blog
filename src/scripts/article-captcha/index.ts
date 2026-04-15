// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { bindSliderInteractions } from "./interactions.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import {
	createFreshCaptchaState,
	createRotateChallenge,
	evaluateRotationAttempt,
	resolveCanvasSize,
	resolveCircleRadius,
	resolveFloatingPanelPosition,
	resolveVisibleCanvasLimit,
	sliderValueToRotation,
} from "./logic.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { loadBackgroundImage, renderCaptchaScene } from "./renderer.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { markGatePassed } from "./state.mjs";

const GATE_SELECTOR = "[data-article-captcha-gate]";
const STORAGE_VALUE = "passed";
const LOCK_CLASS = "article-captcha-locked";
const SUCCESS_DISMISS_DELAY_MS = 500;
const CAPTCHA_INSTRUCTION_TEXT = "拖动滑块完成验证";
const CAPTCHA_SUCCESS_TEXT = "验证成功，正在进入页面...";
const CAPTCHA_RETRY_TEXT = "验证失败，请重试";
const CAPTCHA_LOADING_TEXT = "正在加载验证码...";
const CAPTCHA_LOAD_ERROR_TEXT = "验证码加载失败，请刷新页面重试";

const captchaConfig = Object.freeze({
	rotationToleranceDeg: 6,
	maxCanvasWidth: 760,
	padding: 18,
	circleRadiusRatio: 0.18,
	sliderMinValue: 0,
	sliderMaxValue: 100,
	minTravelTurns: 0.5,
	maxTravelTurns: 0.95,
	targetSliderPaddingRatio: 0.18,
	floatingControlsPadding: 18,
	floatingControlsGap: 18,
});

type GateRoot = HTMLElement & {
	dataset: DOMStringMap & {
		storageKey?: string;
		backgroundImageUrl?: string;
		fallbackBackgroundImageUrl?: string;
		gateState?: string;
	};
};

type ArticleCaptchaElements = {
	content: HTMLElement;
	overlay: HTMLElement;
	card: HTMLElement;
	header: HTMLElement;
	canvasFrame: HTMLElement;
	controls: HTMLElement;
	canvas: HTMLCanvasElement;
	slider: HTMLInputElement;
	status: HTMLElement;
};

type RelativeRect = {
	left: number;
	top: number;
	right: number;
	bottom: number;
};

type ArticleCaptchaController = {
	root: GateRoot;
	destroy: () => void;
};

type SwupHookRegistry = {
	on: (name: string, handler: () => void) => void;
};

type SwupWindow = Window & {
	__articleCaptchaBootstrapped__?: boolean;
	__articleCaptchaSwupHookAttached__?: boolean;
	swup?: {
		hooks?: SwupHookRegistry;
	};
};

const controllers = new Map<HTMLElement, ArticleCaptchaController>();

function isGateOpen(root: GateRoot) {
	return root.dataset.gateState !== "passed";
}

function updateDocumentLockState() {
	const hasOpenGate = Array.from(controllers.values()).some(({ root }) =>
		root.isConnected && isGateOpen(root),
	);
	document.body.classList.toggle(LOCK_CLASS, hasOpenGate);
}

function getSessionStorage() {
	try {
		return sessionStorage;
	} catch {
		return undefined;
	}
}

function readPersistedState(storageKey: string) {
	return getSessionStorage()?.getItem(storageKey) === STORAGE_VALUE;
}

function pause(ms: number) {
	if (ms <= 0) {
		return Promise.resolve();
	}

	return new Promise<void>((resolve) => {
		window.setTimeout(resolve, ms);
	});
}

function updateSliderVisual(slider: HTMLInputElement, value: number) {
	const min = Number(slider.min || 0);
	const max = Number(slider.max || 1);
	const span = Math.max(max - min, 1);
	const percentage = ((value - min) / span) * 100;
	slider.style.setProperty("--range-progress", `${percentage}%`);
}

function setGateLocked(root: GateRoot, elements: ArticleCaptchaElements, locked: boolean) {
	root.dataset.gateState = locked ? "locked" : "passed";
	elements.overlay.hidden = !locked;
	elements.content.inert = locked;
	elements.overlay.setAttribute("aria-hidden", locked ? "false" : "true");
	updateDocumentLockState();
}

function setGatePassed(root: GateRoot, elements: ArticleCaptchaElements, storageKey: string) {
	markGatePassed(
		{
			root,
			overlay: elements.overlay,
			content: elements.content,
		},
		{
			storageKey,
			storage: getSessionStorage(),
		},
	);
	updateDocumentLockState();
}

function getRequiredElement<T extends Element>(root: ParentNode, selector: string) {
	const element = root.querySelector<T>(selector);
	if (!element) {
		throw new Error(`Missing captcha element: ${selector}`);
	}

	return element;
}

function collectElements(root: GateRoot): ArticleCaptchaElements {
	return {
		content: getRequiredElement<HTMLElement>(root, "[data-article-captcha-content]"),
		overlay: getRequiredElement<HTMLElement>(root, "[data-article-captcha-overlay]"),
		card: getRequiredElement<HTMLElement>(root, "[data-article-captcha-card]"),
		header: getRequiredElement<HTMLElement>(root, ".article-captcha-header"),
		canvasFrame: getRequiredElement<HTMLElement>(root, "[data-article-captcha-canvas-frame]"),
		controls: getRequiredElement<HTMLElement>(root, "[data-article-captcha-controls]"),
		canvas: getRequiredElement<HTMLCanvasElement>(root, "[data-article-captcha-canvas]"),
		slider: getRequiredElement<HTMLInputElement>(root, "[data-article-captcha-slider]"),
		status: getRequiredElement<HTMLElement>(root, "[data-article-captcha-status]"),
	};
}

function mountCaptchaGate(root: GateRoot): ArticleCaptchaController {
	const storageKey = root.dataset.storageKey ?? "site-captcha:passed";
	const backgroundImageUrl = root.dataset.backgroundImageUrl ?? "/captcha/preview.jpg";
	const fallbackBackgroundImageUrl =
		root.dataset.fallbackBackgroundImageUrl ?? "/captcha/placeholder-background.svg";
	const elements = collectElements(root);
	const context = elements.canvas.getContext("2d");

	if (!context) {
		throw new Error("Unable to create the canvas rendering context for the site captcha.");
	}

	let backgroundImage: HTMLImageElement | undefined;
	let challenge: ReturnType<typeof createRotateChallenge> | undefined;
	let challengeVersion = 0;
	let controlsPositionFrame = 0;
	let challengeState = createFreshCaptchaState({
		startRotationDeg: captchaConfig.sliderMinValue,
		startSliderValue: captchaConfig.sliderMinValue,
	});

	const clampSliderValue = (value: number) =>
		Math.min(
			Math.max(Number(value), captchaConfig.sliderMinValue),
			captchaConfig.sliderMaxValue,
		);

	const setStatus = (state: "idle" | "loading" | "error" | "success", message: string) => {
		challengeState.status = state;
		elements.status.dataset.state = state;
		elements.status.textContent = message;
	};

	const toRelativeRect = (rect: DOMRect, containerRect: DOMRect): RelativeRect => ({
		left: rect.left - containerRect.left,
		top: rect.top - containerRect.top,
		right: rect.right - containerRect.left,
		bottom: rect.bottom - containerRect.top,
	});

	const readOverlayPadding = () => {
		const overlayStyles = getComputedStyle(elements.overlay);
		const values = [
			overlayStyles.paddingTop,
			overlayStyles.paddingRight,
			overlayStyles.paddingBottom,
			overlayStyles.paddingLeft,
		]
			.map((value) => Number.parseFloat(value))
			.filter((value) => Number.isFinite(value));

		return values.length > 0
			? Math.max(captchaConfig.floatingControlsPadding, ...values)
			: captchaConfig.floatingControlsPadding;
	};

	const readStyleSum = (styles: CSSStyleDeclaration, propertyNames: string[]) =>
		propertyNames.reduce((sum, propertyName) => {
			const value = Number.parseFloat(styles.getPropertyValue(propertyName));
			return Number.isFinite(value) ? sum + value : sum;
		}, 0);

	const readGapValue = (styles: CSSStyleDeclaration) => {
		for (const propertyName of ["row-gap", "gap"]) {
			const value = Number.parseFloat(styles.getPropertyValue(propertyName));
			if (Number.isFinite(value)) {
				return value;
			}
		}

		return 0;
	};

	const applyFloatingControlsPosition = () => {
		if (!root.isConnected || root.dataset.gateState === "passed") {
			return;
		}

		const overlayRect = elements.overlay.getBoundingClientRect();
		const controlsRect = elements.controls.getBoundingClientRect();
		if (
			overlayRect.width <= 0 ||
			overlayRect.height <= 0 ||
			controlsRect.width <= 0 ||
			controlsRect.height <= 0
		) {
			return;
		}

		const padding = readOverlayPadding();
		const candidateBlockedRects = [elements.card, elements.canvasFrame]
			.map((element) => element.getBoundingClientRect())
			.filter((rect) => rect.width > 0 && rect.height > 0)
			.map((rect) => toRelativeRect(rect, overlayRect));

		let nextPosition: { left: number; top: number } | undefined;
		for (const blockedRect of candidateBlockedRects) {
			try {
				nextPosition = resolveFloatingPanelPosition({
					viewportWidth: overlayRect.width,
					viewportHeight: overlayRect.height,
					panelWidth: controlsRect.width,
					panelHeight: controlsRect.height,
					blockedRect,
					padding,
					gap: captchaConfig.floatingControlsGap,
				});
				break;
			} catch {
				continue;
			}
		}

		if (!nextPosition) {
			try {
				nextPosition = resolveFloatingPanelPosition({
					viewportWidth: overlayRect.width,
					viewportHeight: overlayRect.height,
					panelWidth: controlsRect.width,
					panelHeight: controlsRect.height,
					padding,
					gap: 0,
				});
			} catch {
				nextPosition = {
					left: padding,
					top: padding,
				};
			}
		}

		elements.controls.style.setProperty(
			"--article-captcha-controls-left",
			`${nextPosition.left}px`,
		);
		elements.controls.style.setProperty(
			"--article-captcha-controls-top",
			`${nextPosition.top}px`,
		);
		elements.controls.dataset.positioned = "true";
	};

	const scheduleFloatingControlsPosition = () => {
		if (controlsPositionFrame) {
			cancelAnimationFrame(controlsPositionFrame);
		}

		controlsPositionFrame = requestAnimationFrame(() => {
			controlsPositionFrame = 0;
			applyFloatingControlsPosition();
		});
	};

	const drawScene = () => {
		if (!backgroundImage || !challenge) {
			context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
			return;
		}

		renderCaptchaScene({
			canvas: elements.canvas,
			context,
			backgroundImage,
			challenge,
			rotationDeg: challengeState.currentRotationDeg,
			status: challengeState.status,
		});
	};

	const configureCanvasForImage = (image: HTMLImageElement) => {
		const sourceWidth = image.naturalWidth || image.width || captchaConfig.maxCanvasWidth;
		const sourceHeight = image.naturalHeight || image.height || captchaConfig.maxCanvasWidth;
		const overlayRect = elements.overlay.getBoundingClientRect();
		const headerRect = elements.header.getBoundingClientRect();
		const cardStyles = getComputedStyle(elements.card);
		const frameStyles = getComputedStyle(elements.canvasFrame);
		const overlayPadding = readOverlayPadding();
		const cardPaddingX = readStyleSum(cardStyles, ["padding-left", "padding-right"]);
		const cardPaddingY = readStyleSum(cardStyles, ["padding-top", "padding-bottom"]);
		const framePaddingX = readStyleSum(frameStyles, ["padding-left", "padding-right"]);
		const framePaddingY = readStyleSum(frameStyles, ["padding-top", "padding-bottom"]);
		const contentGap = readGapValue(cardStyles);
		const visibleCanvasLimit = resolveVisibleCanvasLimit({
			viewportWidth: overlayRect.width || window.innerWidth,
			viewportHeight: overlayRect.height || window.innerHeight,
			overlayPadding,
			cardPaddingX,
			cardPaddingY,
			headerHeight: headerRect.height,
			contentGap,
			framePaddingX,
			framePaddingY,
			maxCanvasWidth: captchaConfig.maxCanvasWidth,
		});
		const { canvasWidth, canvasHeight } = resolveCanvasSize({
			sourceWidth,
			sourceHeight,
			maxCanvasWidth: visibleCanvasLimit,
		});

		elements.canvas.width = canvasWidth;
		elements.canvas.height = canvasHeight;
		elements.overlay.style.setProperty("--article-captcha-canvas-limit", `${canvasWidth}px`);
	};

	const configureSlider = () => {
		elements.slider.min = String(captchaConfig.sliderMinValue);
		elements.slider.max = String(captchaConfig.sliderMaxValue);
		elements.slider.step = "0.01";
		elements.slider.value = String(challengeState.sliderValue);
		updateSliderVisual(elements.slider, challengeState.sliderValue);
	};

	const syncRotation = (sliderValue: number) => {
		if (!challenge) {
			return;
		}

		const nextSliderValue = clampSliderValue(sliderValue);
		const nextRotationDeg = sliderValueToRotation({
			sliderValue: nextSliderValue,
			challenge,
		});

		challengeState.currentRotationDeg = nextRotationDeg;
		challengeState.sliderValue = nextSliderValue;
		elements.slider.value = String(nextSliderValue);
		updateSliderVisual(elements.slider, nextSliderValue);
		drawScene();
	};

	const animateBackToStart = (version: number) =>
		new Promise<boolean>((resolve) => {
			const fromValue = challengeState.sliderValue;
			const toValue = challengeState.startSliderValue;
			const duration = 320;
			const startTime = performance.now();

			const step = (now: number) => {
				if (version !== challengeVersion) {
					resolve(false);
					return;
				}

				const progress = Math.min((now - startTime) / duration, 1);
				const eased = 1 - (1 - progress) ** 3;
				const nextSliderValue = fromValue + (toValue - fromValue) * eased;
				syncRotation(nextSliderValue);

				if (progress < 1) {
					requestAnimationFrame(step);
					return;
				}

				syncRotation(toValue);
				resolve(true);
			};

			requestAnimationFrame(step);
		});

	const sliderController = bindSliderInteractions({
		slider: elements.slider,
		onMove(value: number) {
			if (!challenge || challengeState.isAnimating || challengeState.isLocked) {
				return;
			}

			setStatus("idle", CAPTCHA_INSTRUCTION_TEXT);
			syncRotation(value);
		},
		async onRelease(value: number) {
			if (!challenge || challengeState.isAnimating || challengeState.isLocked) {
				return;
			}

			syncRotation(value);

			const result = evaluateRotationAttempt({
				currentRotationDeg: challengeState.currentRotationDeg,
				targetRotationDeg: challenge.targetRotationDeg,
				toleranceDeg: captchaConfig.rotationToleranceDeg,
			});

			if (result.success) {
				challengeState.isLocked = true;
				const successVersion = challengeVersion;
				sliderController.setDisabled(true);
				setStatus("success", CAPTCHA_SUCCESS_TEXT);
				drawScene();
				await pause(SUCCESS_DISMISS_DELAY_MS);
				if (successVersion !== challengeVersion || !root.isConnected) {
					return;
				}
				setGatePassed(root, elements, storageKey);
				return;
			}

			const releaseVersion = challengeVersion;
			challengeState.isAnimating = true;
			sliderController.setDisabled(true);
			setStatus("error", CAPTCHA_RETRY_TEXT);
			drawScene();

			const completed = await animateBackToStart(releaseVersion);

			if (!completed || releaseVersion !== challengeVersion) {
				return;
			}

			challengeState.isAnimating = false;
			sliderController.setDisabled(false);
			setStatus("error", CAPTCHA_RETRY_TEXT);
			drawScene();
		},
	});

	const createChallenge = () => {
		challengeVersion += 1;
		const circleRadius = resolveCircleRadius({
			canvasWidth: elements.canvas.width,
			canvasHeight: elements.canvas.height,
			padding: captchaConfig.padding,
			circleRadiusRatio: captchaConfig.circleRadiusRatio,
		});

		challenge = createRotateChallenge({
			canvasWidth: elements.canvas.width,
			canvasHeight: elements.canvas.height,
			circleRadius,
			padding: captchaConfig.padding,
			sliderMinValue: captchaConfig.sliderMinValue,
			sliderMaxValue: captchaConfig.sliderMaxValue,
			minTravelTurns: captchaConfig.minTravelTurns,
			maxTravelTurns: captchaConfig.maxTravelTurns,
			targetSliderPaddingRatio: captchaConfig.targetSliderPaddingRatio,
		});

		challengeState = createFreshCaptchaState({
			startRotationDeg: challenge.startRotationDeg,
			startSliderValue: challenge.startSliderValue,
		});
		sliderController.clearPointerSession();
		configureSlider();
		sliderController.setDisabled(false);
		setStatus("idle", CAPTCHA_INSTRUCTION_TEXT);
		drawScene();
		scheduleFloatingControlsPosition();
		elements.slider.focus({ preventScroll: true });
	};

	const initializeCaptcha = async () => {
		if (readPersistedState(storageKey)) {
			setGatePassed(root, elements, storageKey);
			return;
		}

		setGateLocked(root, elements, true);
		sliderController.setDisabled(true);
		updateSliderVisual(elements.slider, captchaConfig.sliderMinValue);
		setStatus("loading", CAPTCHA_LOADING_TEXT);
		scheduleFloatingControlsPosition();

		try {
			const imageState = await loadBackgroundImage(
				backgroundImageUrl,
				fallbackBackgroundImageUrl,
			);

			backgroundImage = imageState.image;
			configureCanvasForImage(backgroundImage);
			createChallenge();
		} catch (error) {
			sliderController.setDisabled(true);
			console.error("Unable to initialize the site captcha background.", error);
			setStatus("error", CAPTCHA_LOAD_ERROR_TEXT);
		}
	};

	const handleViewportResize = () => {
		scheduleFloatingControlsPosition();
	};

	window.addEventListener("resize", handleViewportResize);
	void initializeCaptcha();

	return {
		root,
		destroy() {
			if (controlsPositionFrame) {
				cancelAnimationFrame(controlsPositionFrame);
			}
			sliderController.destroy();
			window.removeEventListener("resize", handleViewportResize);
			if (root.isConnected) {
				elements.content.inert = false;
			}
		},
	};
}

function cleanupDisconnectedControllers() {
	for (const [root, controller] of controllers) {
		if (root.isConnected) {
			continue;
		}

		controller.destroy();
		controllers.delete(root);
	}
}

export function initializeArticleCaptchaGates(scope: ParentNode = document) {
	cleanupDisconnectedControllers();

	const gates = scope.querySelectorAll<GateRoot>(GATE_SELECTOR);
	for (const gate of gates) {
		if (controllers.has(gate)) {
			continue;
		}

		controllers.set(gate, mountCaptchaGate(gate));
	}

	updateDocumentLockState();
}

function attachSwupHook() {
	const runtimeWindow = window as SwupWindow;
	if (runtimeWindow.__articleCaptchaSwupHookAttached__) {
		return;
	}

	const attach = () => {
		if (!runtimeWindow.swup?.hooks || runtimeWindow.__articleCaptchaSwupHookAttached__) {
			return;
		}

		runtimeWindow.swup.hooks.on("page:view", () => initializeArticleCaptchaGates());
		runtimeWindow.__articleCaptchaSwupHookAttached__ = true;
	};

	attach();

	if (!runtimeWindow.__articleCaptchaSwupHookAttached__) {
		document.addEventListener("swup:enable", attach, { once: true });
	}
}

const runtimeWindow = window as SwupWindow;
if (!runtimeWindow.__articleCaptchaBootstrapped__) {
	runtimeWindow.__articleCaptchaBootstrapped__ = true;

	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", () => initializeArticleCaptchaGates(), {
			once: true,
		});
	} else {
		initializeArticleCaptchaGates();
	}

	attachSwupHook();
}
