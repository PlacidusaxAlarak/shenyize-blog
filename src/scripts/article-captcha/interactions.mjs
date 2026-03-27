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
		},
		destroy() {
			pointerActive = false;
			slider.removeEventListener("input", handleInput);
			slider.removeEventListener("pointerdown", markPointerActive);
			slider.removeEventListener("mousedown", markPointerActive);
			slider.removeEventListener("touchstart", markPointerActive);
			slider.removeEventListener("keyup", releaseFromKeyboard);
			document.removeEventListener("pointerup", releasePointer);
			document.removeEventListener("mouseup", releasePointer);
			document.removeEventListener("touchend", releasePointer);
			document.removeEventListener("touchcancel", releasePointer);
		},
	};
}
