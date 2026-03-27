export interface SliderInteractionController {
	setDisabled(disabled: boolean): void;
	clearPointerSession(): void;
	destroy(): void;
}

export function bindSliderInteractions(options: {
	slider: HTMLInputElement;
	onMove: (value: number) => void;
	onRelease: (value: number) => void | Promise<void>;
}): SliderInteractionController;
