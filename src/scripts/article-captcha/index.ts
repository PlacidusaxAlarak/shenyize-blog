// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { bindSliderInteractions } from "./interactions.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import {
	createFreshCaptchaState,
	createRotateChallenge,
	evaluateRotationAttempt,
	resolveCanvasSize,
	resolveCircleRadius,
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

const captchaConfig = Object.freeze({
	rotationToleranceDeg: 6,
	maxCanvasWidth: 620,
	padding: 18,
	circleRadiusRatio: 0.18,
	sliderMinValue: 0,
	sliderMaxValue: 100,
	minTravelTurns: 0.5,
	maxTravelTurns: 0.95,
	targetSliderPaddingRatio: 0.18,
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
	canvas: HTMLCanvasElement;
	slider: HTMLInputElement;
	refreshButton: HTMLButtonElement;
	status: HTMLElement;
	meta: HTMLElement;
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
		canvas: getRequiredElement<HTMLCanvasElement>(root, "[data-article-captcha-canvas]"),
		slider: getRequiredElement<HTMLInputElement>(root, "[data-article-captcha-slider]"),
		refreshButton: getRequiredElement<HTMLButtonElement>(root, "[data-article-captcha-refresh]"),
		status: getRequiredElement<HTMLElement>(root, "[data-article-captcha-status]"),
		meta: getRequiredElement<HTMLElement>(root, "[data-article-captcha-meta]"),
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
	let usingFallbackBackground = false;
	let challengeVersion = 0;
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

	const updateMeta = () => {
		const parts = [];

		if (usingFallbackBackground) {
			parts.push("主图加载失败，当前使用本站占位图。");
		}

		parts.push(
			`每道题的正确位置和旋转灵敏度都会随机变化，通过条件为角度误差不超过 ±${captchaConfig.rotationToleranceDeg}°。`,
		);

		elements.meta.textContent = parts.join("");
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
		const { canvasWidth, canvasHeight } = resolveCanvasSize({
			sourceWidth,
			sourceHeight,
			maxCanvasWidth: captchaConfig.maxCanvasWidth,
		});

		elements.canvas.width = canvasWidth;
		elements.canvas.height = canvasHeight;
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

			setStatus("idle", "继续旋转圆片，直到圆内图像与外部背景完全贴合。");
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
				elements.refreshButton.disabled = true;
				setStatus(
					"success",
					`验证成功，当前角度误差 ${Math.round(result.deltaDeg)}°。正在继续显示页面...`,
				);
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
			setStatus(
				"error",
				`验证失败，当前角度误差 ${Math.round(result.deltaDeg)}°。圆片正在回到起始位置。`,
			);
			drawScene();

			const completed = await animateBackToStart(releaseVersion);

			if (!completed || releaseVersion !== challengeVersion) {
				return;
			}

			challengeState.isAnimating = false;
			sliderController.setDisabled(false);
			setStatus("error", "验证失败，圆片已回到起始位置。你可以重试，或点击“刷新验证码”。");
			drawScene();
		},
	});

	const createChallenge = (reason: "initial" | "refresh") => {
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
		elements.refreshButton.disabled = false;
		updateMeta();
		setStatus(
			"idle",
			reason === "refresh"
				? "验证码已刷新，正确位置和旋转灵敏度已重新生成。"
				: "拖动下方滑块，旋转中央圆片，让图像重新对齐。",
		);
		drawScene();
		elements.slider.focus();
	};

	const initializeCaptcha = async () => {
		if (readPersistedState(storageKey)) {
			setGatePassed(root, elements, storageKey);
			return;
		}

		setGateLocked(root, elements, true);
		sliderController.setDisabled(true);
		elements.refreshButton.disabled = true;
		updateSliderVisual(elements.slider, captchaConfig.sliderMinValue);
		setStatus("loading", "正在加载验证码...");

		try {
			const imageState = await loadBackgroundImage(
				backgroundImageUrl,
				fallbackBackgroundImageUrl,
			);

			backgroundImage = imageState.image;
			usingFallbackBackground = imageState.usedFallback;
			configureCanvasForImage(backgroundImage);
			createChallenge("initial");
		} catch (error) {
			sliderController.setDisabled(true);
			elements.refreshButton.disabled = true;
			elements.meta.textContent = "背景图加载失败，请检查验证码图片路径。";
			setStatus("error", error instanceof Error ? error.message : String(error));
		}
	};

	const handleRefresh = () => {
		if (!backgroundImage) {
			return;
		}

		createChallenge("refresh");
	};

	elements.refreshButton.addEventListener("click", handleRefresh);
	void initializeCaptcha();

	return {
		root,
		destroy() {
			sliderController.destroy();
			elements.refreshButton.removeEventListener("click", handleRefresh);
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
