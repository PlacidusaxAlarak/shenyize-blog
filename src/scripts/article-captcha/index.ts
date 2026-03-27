// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { bindSliderInteractions } from "./interactions.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { createChallengeGeometry, createFreshCaptchaState, evaluateAttempt, sliderValueToPieceX } from "./logic.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { loadBackgroundImage, renderCaptchaScene } from "./renderer.mjs";
// @ts-ignore This client bundle imports runtime-authored .mjs helpers for direct node:test coverage.
import { markGatePassed } from "./state.mjs";

const GATE_SELECTOR = "[data-article-captcha-gate]";
const STORAGE_VALUE = "passed";
const LOCK_CLASS = "article-captcha-locked";
const SUCCESS_DISMISS_DELAY_MS = 500;

const captchaConfig = Object.freeze({
	tolerancePx: 5,
	canvasWidth: 360,
	canvasHeight: 220,
	sliderStartX: 24,
	pieceRadius: 34,
	padding: 18,
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

function readSessionState(storageKey: string) {
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
	const max = Number(slider.max || 1);
	const percentage = max === 0 ? 0 : (value / max) * 100;
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
	const storageKey = root.dataset.storageKey ?? "article-captcha:posts";
	const backgroundImageUrl = root.dataset.backgroundImageUrl ?? "/captcha/demo-background.svg";
	const fallbackBackgroundImageUrl =
		root.dataset.fallbackBackgroundImageUrl ?? "/captcha/placeholder-background.svg";
	const elements = collectElements(root);
	const context = elements.canvas.getContext("2d");

	if (!context) {
		throw new Error("Unable to create the canvas rendering context for the article captcha.");
	}

	elements.canvas.width = captchaConfig.canvasWidth;
	elements.canvas.height = captchaConfig.canvasHeight;

	let backgroundImage: HTMLImageElement | undefined;
	let usingFallbackBackground = false;
	let geometry = createChallengeGeometry(captchaConfig);
	let challengeVersion = 0;
	let challengeState = createFreshCaptchaState({ sliderStartX: captchaConfig.sliderStartX });

	const setStatus = (state: "idle" | "loading" | "error" | "success", message: string) => {
		challengeState.status = state;
		elements.status.dataset.state = state;
		elements.status.textContent = message;
	};

	const updateMeta = () => {
		elements.meta.textContent = usingFallbackBackground
			? "主背景加载失败，当前使用站内占位图。题面里包含 1 个真实槽位和 1 个迷惑槽位。"
			: "当前题面包含 1 个真实槽位和 1 个迷惑槽位；点击“刷新验证码”可重新生成位置和角度。";
	};

	const drawScene = () => {
		if (!backgroundImage) {
			context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
			return;
		}

		renderCaptchaScene({
			canvas: elements.canvas,
			context,
			backgroundImage,
			geometry,
			pieceX: challengeState.currentPieceX,
			status: challengeState.status,
		});
	};

	const configureSlider = () => {
		elements.slider.min = "0";
		elements.slider.max = String(geometry.maxTravel);
		elements.slider.step = "0.1";
		elements.slider.value = String(challengeState.sliderValue);
		updateSliderVisual(elements.slider, challengeState.sliderValue);
	};

	const syncPiecePosition = (sliderValue: number) => {
		const nextPieceX = sliderValueToPieceX({
			sliderValue,
			sliderStartX: geometry.sliderStartX,
			maxTravel: geometry.maxTravel,
		});

		challengeState.currentPieceX = nextPieceX;
		challengeState.sliderValue = nextPieceX - geometry.sliderStartX;
		elements.slider.value = String(challengeState.sliderValue);
		updateSliderVisual(elements.slider, challengeState.sliderValue);
		drawScene();
	};

	const animateBackToStart = (version: number) =>
		new Promise<boolean>((resolve) => {
			const fromX = challengeState.currentPieceX;
			const toX = geometry.sliderStartX;
			const delta = fromX - toX;
			const duration = 320;
			const startTime = performance.now();

			const step = (now: number) => {
				if (version !== challengeVersion) {
					resolve(false);
					return;
				}

				const progress = Math.min((now - startTime) / duration, 1);
				const eased = 1 - (1 - progress) ** 3;
				challengeState.currentPieceX = fromX - delta * eased;
				challengeState.sliderValue = challengeState.currentPieceX - geometry.sliderStartX;
				elements.slider.value = String(challengeState.sliderValue);
				updateSliderVisual(elements.slider, challengeState.sliderValue);
				drawScene();

				if (progress < 1) {
					requestAnimationFrame(step);
					return;
				}

				challengeState.currentPieceX = toX;
				challengeState.sliderValue = 0;
				elements.slider.value = "0";
				updateSliderVisual(elements.slider, 0);
				drawScene();
				resolve(true);
			};

			requestAnimationFrame(step);
		});

	const sliderController = bindSliderInteractions({
		slider: elements.slider,
		onMove(value: number) {
			if (challengeState.isAnimating || challengeState.isLocked) {
				return;
			}

			setStatus("idle", "继续拖动滑块，让拼块对准真实槽位；另一处缺口只是迷惑项。");
			syncPiecePosition(value);
		},
		async onRelease(value: number) {
			if (challengeState.isAnimating || challengeState.isLocked) {
				return;
			}

			syncPiecePosition(value);

			const result = evaluateAttempt({
				pieceX: challengeState.currentPieceX,
				targetX: geometry.targetX,
				tolerancePx: captchaConfig.tolerancePx,
			});

			if (result.success) {
				challengeState.isLocked = true;
				const successVersion = challengeVersion;
				sliderController.setDisabled(true);
				elements.refreshButton.disabled = true;
				setStatus("success", `验证成功，当前误差 ${Math.round(result.delta)}px。正在为你继续显示正文...`);
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
			setStatus("error", `验证失败，当前偏差 ${Math.round(result.delta)}px。你可能对准了迷惑槽位。`);
			drawScene();

			const completed = await animateBackToStart(releaseVersion);

			if (!completed || releaseVersion !== challengeVersion) {
				return;
			}

			challengeState.isAnimating = false;
			sliderController.setDisabled(false);
			setStatus("error", "验证失败，滑块已回到起点。你可以重试，或点击“刷新验证码”。");
			drawScene();
		},
	});

	const createChallenge = (reason: "initial" | "refresh") => {
		challengeVersion += 1;
		geometry = createChallengeGeometry(captchaConfig);
		challengeState = createFreshCaptchaState({ sliderStartX: geometry.sliderStartX });
		sliderController.clearPointerSession();
		configureSlider();
		sliderController.setDisabled(false);
		elements.refreshButton.disabled = false;
		updateMeta();
		setStatus(
			"idle",
			reason === "refresh"
				? "验证码已刷新。真实槽位和迷惑槽位的位置、角度都已重新生成。"
				: "拖动下方滑块，让五边形拼块回到真实目标槽位。",
		);
		drawScene();
		elements.slider.focus();
	};

	const initializeCaptcha = async () => {
		if (readSessionState(storageKey)) {
			setGatePassed(root, elements, storageKey);
			return;
		}

		setGateLocked(root, elements, true);
		sliderController.setDisabled(true);
		elements.refreshButton.disabled = true;
		updateSliderVisual(elements.slider, 0);
		setStatus("loading", "正在加载背景图与验证码画布...");

		try {
			const imageState = await loadBackgroundImage(backgroundImageUrl, fallbackBackgroundImageUrl);

			backgroundImage = imageState.image;
			usingFallbackBackground = imageState.usedFallback;
			createChallenge("initial");
		} catch (error) {
			sliderController.setDisabled(true);
			elements.refreshButton.disabled = true;
			elements.meta.textContent = "背景资源加载失败，请检查验证码图片路径。";
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
