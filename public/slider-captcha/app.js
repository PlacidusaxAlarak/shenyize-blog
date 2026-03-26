import { bindSliderInteractions } from "./js/captcha-interactions.js";
import {
  createChallengeGeometry,
  createFreshCaptchaState,
  evaluateAttempt,
  sliderValueToPieceX
} from "./js/captcha-logic.js";
import { captchaConfig } from "./js/config.js";
import { loadBackgroundImage, renderCaptchaScene } from "./js/captcha-renderer.js";

const canvas = document.querySelector("#captcha-canvas");
const slider = document.querySelector("#captcha-slider");
const refreshButton = document.querySelector("#captcha-refresh");
const statusElement = document.querySelector("#captcha-status");
const metaElement = document.querySelector("#captcha-meta");
const context = canvas.getContext("2d");

canvas.width = captchaConfig.canvasWidth;
canvas.height = captchaConfig.canvasHeight;

let backgroundImage;
let geometry;
let usingFallbackBackground = false;
let challengeVersion = 0;
let challengeState = createFreshCaptchaState({ sliderStartX: captchaConfig.sliderStartX });

function updateSliderVisual(value) {
  const max = Number(slider.max || 1);
  const percentage = max === 0 ? 0 : (value / max) * 100;
  slider.style.setProperty("--range-progress", `${percentage}%`);
}

function setStatus(state, message) {
  challengeState.status = state;
  statusElement.dataset.state = state;
  statusElement.textContent = message;
}

function updateMeta() {
  metaElement.textContent = usingFallbackBackground
    ? "主背景加载失败，当前使用本地占位图。题面里包含 1 个真实槽位和 1 个迷惑槽位。"
    : "当前题面包含 1 个真实槽位和 1 个迷惑槽位；点击“刷新验证码”会重新生成位置和倾斜角度。";
}

function drawScene() {
  if (!backgroundImage || !geometry) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  renderCaptchaScene({
    canvas,
    context,
    backgroundImage,
    geometry,
    pieceX: challengeState.currentPieceX,
    status: challengeState.status
  });
}

function configureSlider() {
  slider.min = "0";
  slider.max = String(geometry.maxTravel);
  slider.step = "0.1";
  slider.value = String(challengeState.sliderValue);
  updateSliderVisual(challengeState.sliderValue);
}

function syncPiecePosition(sliderValue) {
  const nextPieceX = sliderValueToPieceX({
    sliderValue,
    sliderStartX: geometry.sliderStartX,
    maxTravel: geometry.maxTravel
  });

  challengeState.currentPieceX = nextPieceX;
  challengeState.sliderValue = nextPieceX - geometry.sliderStartX;
  slider.value = String(challengeState.sliderValue);
  updateSliderVisual(challengeState.sliderValue);
  drawScene();
}

function createChallenge(reason) {
  challengeVersion += 1;
  geometry = createChallengeGeometry({
    canvasWidth: captchaConfig.canvasWidth,
    canvasHeight: captchaConfig.canvasHeight,
    pieceRadius: captchaConfig.pieceRadius,
    sliderStartX: captchaConfig.sliderStartX,
    padding: captchaConfig.padding
  });

  challengeState = createFreshCaptchaState({ sliderStartX: geometry.sliderStartX });
  sliderController.clearPointerSession();
  configureSlider();
  sliderController.setDisabled(false);
  refreshButton.disabled = false;
  updateMeta();
  setStatus(
    "idle",
    reason === "refresh"
      ? "验证码已刷新。真实槽位和迷惑槽位的位置、角度都已重新生成。"
      : "拖动下方滑块，让五边形拼块回到真实目标槽位。"
  );
  drawScene();
}

function animateBackToStart(version) {
  return new Promise((resolve) => {
    const fromX = challengeState.currentPieceX;
    const toX = geometry.sliderStartX;
    const delta = fromX - toX;
    const duration = 320;
    const startTime = performance.now();

    const step = (now) => {
      if (version !== challengeVersion) {
        resolve(false);
        return;
      }

      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - (1 - progress) ** 3;
      challengeState.currentPieceX = fromX - delta * eased;
      challengeState.sliderValue = challengeState.currentPieceX - geometry.sliderStartX;
      slider.value = String(challengeState.sliderValue);
      updateSliderVisual(challengeState.sliderValue);
      drawScene();

      if (progress < 1) {
        requestAnimationFrame(step);
        return;
      }

      challengeState.currentPieceX = toX;
      challengeState.sliderValue = 0;
      slider.value = "0";
      updateSliderVisual(0);
      drawScene();
      resolve(true);
    };

    requestAnimationFrame(step);
  });
}

const sliderController = bindSliderInteractions({
  slider,
  onMove(value) {
    if (!geometry || challengeState.isAnimating || challengeState.isLocked) {
      return;
    }

    setStatus("idle", "继续拖动滑块，让拼块对准真实槽位；另一个槽位只是迷惑项。");
    syncPiecePosition(value);
  },
  async onRelease(value) {
    if (!geometry || challengeState.isAnimating || challengeState.isLocked) {
      return;
    }

    syncPiecePosition(value);

    const result = evaluateAttempt({
      pieceX: challengeState.currentPieceX,
      targetX: geometry.targetX,
      tolerancePx: captchaConfig.tolerancePx
    });

    if (result.success) {
      challengeState.isLocked = true;
      sliderController.setDisabled(true);
      setStatus("success", `验证成功，当前误差 ${Math.round(result.delta)}px。可点击“刷新验证码”生成新题。`);
      drawScene();
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
  }
});

refreshButton.addEventListener("click", () => {
  if (!backgroundImage) {
    return;
  }

  createChallenge("refresh");
});

async function initializeCaptcha() {
  sliderController.setDisabled(true);
  refreshButton.disabled = true;
  updateSliderVisual(0);
  setStatus("loading", "正在加载背景图与验证码画布...");

  try {
    const imageState = await loadBackgroundImage(
      captchaConfig.backgroundImageUrl,
      captchaConfig.fallbackBackgroundImageUrl
    );

    backgroundImage = imageState.image;
    usingFallbackBackground = imageState.usedFallback;
    createChallenge("initial");
  } catch (error) {
    sliderController.setDisabled(true);
    refreshButton.disabled = true;
    metaElement.textContent = "背景资源加载失败，请检查 public/js/config.js 中的图片路径。";
    setStatus("error", error instanceof Error ? error.message : String(error));
  }
}

initializeCaptcha();
