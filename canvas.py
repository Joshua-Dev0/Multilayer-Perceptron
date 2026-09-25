
import os
import tkinter as tk
import cupy as cp

from multilayer_perceptron3 import forward_propagation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "model_55K.npz")

model = cp.load(model_path)

W1 = model["W1"]
B1 = model["B1"]

W2 = model["W2"]
B2 = model["B2"]

W3 = model["W3"]
B3 = model["B3"]


class NativeMNISTCanvas:

    def __init__(self, root):

        self.root = root
        self.root.title("28x28 AI Digit Recognizer")

        self.grid_size = 28
        self.pixel_scale = 16
        self.canvas_dim = self.grid_size * self.pixel_scale

        # Pixel grid stored on GPU
        self.pixel_grid = cp.zeros(
            (self.grid_size, self.grid_size),
            dtype=cp.float32
        )

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_dim,
            height=self.canvas_dim,
            bg="black"
        )

        self.canvas.pack(pady=10)

        self.draw_grid_lines()

        self.prediction_label = tk.Label(
            root,
            text="Prediction: -",
            font=("Arial", 15)
        )

        self.prediction_label.pack(pady=10)

        # Drawing events
        self.canvas.bind(
            "<B1-Motion>",
            self.paint_brush
        )

        self.canvas.bind(
            "<Button-1>",
            self.paint_brush
        )

        # Buttons
        self.btn_process = tk.Button(
            root,
            text="Inference",
            command=self.output_vector
        )

        self.btn_process.pack(
            side=tk.LEFT,
            padx=30,
            pady=10
        )

        self.btn_clear = tk.Button(
            root,
            text="Clear Canvas",
            command=self.clear_canvas
        )

        self.btn_clear.pack(
            side=tk.RIGHT,
            padx=30,
            pady=10
        )

    def draw_grid_lines(self):

        for i in range(self.grid_size):

            self.canvas.create_line(
                0,
                i * self.pixel_scale,
                self.canvas_dim,
                i * self.pixel_scale,
                fill="#121212"
            )

            self.canvas.create_line(
                i * self.pixel_scale,
                0,
                i * self.pixel_scale,
                self.canvas_dim,
                fill="#121212"
            )

    def paint_brush(self, event):

        col = event.x // self.pixel_scale
        row = event.y // self.pixel_scale

        # Ignore drawing outside the canvas
        if not (
            0 <= row < self.grid_size
            and 0 <= col < self.grid_size
        ):
            return

        for dr in [-1, 0, 1]:

            for dc in [-1, 0, 1]:

                r_target = row + dr
                c_target = col + dc

                if not (
                    0 <= r_target < self.grid_size
                    and 0 <= c_target < self.grid_size
                ):
                    continue

                # Brush intensity
                if dr == 0 and dc == 0:

                    intensity = 255.0

                elif abs(dr) == 1 and abs(dc) == 1:

                    intensity = 100.0

                else:

                    intensity = 180.0

                # Read current GPU pixel
                current = float(
                    self.pixel_grid[
                        r_target,
                        c_target
                    ].get()
                )

                intensity = max(
                    current,
                    intensity
                )

                # Update GPU pixel
                self.pixel_grid[
                    r_target,
                    c_target
                ] = intensity

                # Convert intensity to Tkinter color
                hex_value = int(intensity)

                hex_color = (
                    f"#{hex_value:02x}"
                    f"{hex_value:02x}"
                    f"{hex_value:02x}"
                )

                x1 = c_target * self.pixel_scale
                y1 = r_target * self.pixel_scale

                x2 = x1 + self.pixel_scale
                y2 = y1 + self.pixel_scale

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=hex_color,
                    outline="#121212"
                )

    def clear_canvas(self):

        self.canvas.delete("all")

        self.pixel_grid = cp.zeros(
            (self.grid_size, self.grid_size),
            dtype=cp.float32
        )

        self.prediction_label.config(
            text="Prediction: -"
        )

        self.draw_grid_lines()

    def output_vector(self):

        # Flatten 28x28 image
        flat_array = self.pixel_grid.flatten()

        # Shape: (784, 1)
        X = flat_array.reshape(784, 1)

        # Normalize pixels from 0-255 to 0-1
        X = X / 255.0

        # Forward propagation
        A3, Z3, A2, Z2, A1, Z1 = forward_propagation(
            X,
            W1, B1,
            W2, B2,
            W3, B3
        )

        digit = int(cp.argmax(A3).get())

        confidence = float(A3[digit, 0].get()) * 100

        self.prediction_label.config(
            text=f"Prediction: {digit} ({confidence:.2f}%)"
        )

        global native_mnist_input

        native_mnist_input = X

native_mnist_input = None

root = tk.Tk()

app = NativeMNISTCanvas(root)

root.mainloop()