# 2026-09-18 18:20 KST
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import tkinter as tk
from PIL import Image, ImageDraw

from inference import load_model, preprocess_image, predict

CANVAS_SIZE = 280
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model", "weights.pth")


class HandwritingApp:
    def __init__(self, root):
        self.model = load_model(WEIGHTS_PATH)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_draw)

        self.last_x = None
        self.last_y = None

        self.result_label = tk.Label(root, text="숫자를 그려 주세요", font=("Arial", 16))
        self.result_label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack()
        tk.Button(button_frame, text="인식", command=self.on_recognize).pack(side="left")
        tk.Button(button_frame, text="지우기", command=self.on_clear).pack(side="left")

    def on_click(self, event):
        r = 8
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.last_x, self.last_y = x, y

    def on_draw(self, event):
        x, y = event.x, event.y
        if self.last_x is None or self.last_y is None:
            r = 8
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        else:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill="white", width=16, capstyle=tk.ROUND, smooth=True,
            )
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=16)
        self.last_x, self.last_y = x, y

    def on_clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="숫자를 그려 주세요")

    def on_recognize(self):
        if self.image.getextrema() == (0, 0):
            self.result_label.config(text="그림을 먼저 그려 주세요")
            return
        tensor = preprocess_image(self.image)
        digit, probs = predict(self.model, tensor)
        confidence = probs[digit] * 100
        self.result_label.config(text=f"예측: {digit} ({confidence:.1f}%)")


def main():
    root = tk.Tk()
    root.title("손글씨 숫자 인식")
    HandwritingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
