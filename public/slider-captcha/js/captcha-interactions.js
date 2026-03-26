export function bindSliderInteractions({ slider, onMove, onRelease }) {
  let pointerActive = false;

  const markPointerActive = () => {
    pointerActive = true;
  };

  const handleInput = () => {
    onMove(Number(slider.value));
  };

  const releasePointer = () => {
    if (!pointerActive) {
      return;
    }

    pointerActive = false;
    onRelease(Number(slider.value));
  };

  const releaseFromKeyboard = (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
      onRelease(Number(slider.value));
    }
  };

  slider.addEventListener("input", handleInput);
  slider.addEventListener("pointerdown", markPointerActive);
  slider.addEventListener("mousedown", markPointerActive);
  slider.addEventListener("touchstart", markPointerActive, { passive: true });
  slider.addEventListener("keyup", releaseFromKeyboard);
  document.addEventListener("pointerup", releasePointer);
  document.addEventListener("mouseup", releasePointer);
  document.addEventListener("touchend", releasePointer, { passive: true });
  document.addEventListener("touchcancel", releasePointer, { passive: true });

  return {
    setDisabled(disabled) {
      slider.disabled = disabled;
    },
    clearPointerSession() {
      pointerActive = false;
    }
  };
}
