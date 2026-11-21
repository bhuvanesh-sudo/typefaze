import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

function drawStroke(canvas: HTMLCanvasElement) {
  canvas.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 256, height: 256, right: 256, bottom: 256, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  fireEvent.pointerDown(canvas, { clientX: 128, clientY: 70, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: 128, clientY: 120, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: 128, clientY: 190, pointerId: 1 });
  fireEvent.pointerUp(canvas, { clientX: 128, clientY: 190, pointerId: 1 });
}

describe("App reference input", () => {
  it("keeps Generate disabled until all five letters are drawn", () => {
    render(<App />);
    const generate = screen.getByRole("button", { name: /generate/i });
    expect(generate).toBeDisabled();
    for (const letter of ["H", "O", "n", "a"]) drawStroke(screen.getByLabelText(`Draw ${letter}`) as HTMLCanvasElement);
    expect(generate).toBeDisabled();
    expect(screen.getByText("Still to draw: g")).toBeInTheDocument();
    drawStroke(screen.getByLabelText("Draw g") as HTMLCanvasElement);
    expect(generate).toBeEnabled();
  });
});
