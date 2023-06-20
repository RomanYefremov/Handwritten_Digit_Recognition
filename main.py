import tkinter as tk
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np


class HandwrittenDigitRecognition:
    def __init__(self):
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model('Handwritten_Digit_Recognition/my_model')
        except Exception as e:
            print(f"Error loading the model: {e}")
            exit()

        # Create a blank image for drawing
        self.image_size = (280, 280)
        self.image = Image.new("L", self.image_size, color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Create the GUI window
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognition")

        # Create the drawing canvas
        self.canvas = tk.Canvas(self.window, width=self.image_size[0], height=self.image_size[1], bg='black')
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        # Create the histogram sidebar
        self.histogram_frame = tk.Frame(self.window, width=100, bg='white')
        self.histogram_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.histogram_canvas = tk.Canvas(self.histogram_frame, width=100, height=280, bg='white')
        self.histogram_canvas.pack()

        # Create a label to display the predicted digit
        self.prediction_label = tk.Label(self.window, text="Prediction: ")
        self.prediction_label.pack(side=tk.TOP)

        # Create the Clear button
        self.clear_button = tk.Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.BOTTOM)

    def clear_canvas(self):
        self.draw.rectangle([(0, 0), self.image_size], fill=0)
        self.canvas.delete("all")
        self.update_histogram([])
        self.prediction_label.config(text="Prediction: ")

    def update_histogram(self, prediction):
        self.histogram_canvas.delete("all")
        if len(prediction) > 0:
            max_value = max(prediction)
            bar_height = 15
            max_bar_width = 50
            for i in range(10):
                # Ensure the prediction value is not negative
                bar_value = max(0, prediction[i])

                # Calculate the mismatch value as the difference between the prediction and the index value
                mismatch = abs(i - prediction.argmax())

                # Set the fill color based on the mismatch value
                fill_color = "red" if mismatch > 0 else "blue"

                bar_width = int(bar_value * max_bar_width / max_value)
                bar_x = 10
                bar_y = 10 + i * (bar_height + 10)

                self.histogram_canvas.create_rectangle(
                    bar_x, bar_y, bar_x + bar_width, bar_y + bar_height,
                    fill=fill_color, outline=""
                )

                self.histogram_canvas.create_text(
                    bar_x + bar_width + 5, bar_y + bar_height // 2,
                    text=f"{i}.", font="Arial 8", fill="black"
                )

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        self.draw.line([(x, y), (x + 10, y + 10)], fill=255, width=5)
        self.canvas.create_rectangle(x, y, x + 10, y + 10, fill='white', width=0)
        self.update_prediction()

    def update_prediction(self):
        try:
            # Resize the image to 28x28 pixels
            resized_image = self.image.resize((28, 28))

            # Convert the image to grayscale and normalize the pixel values
            grayscale_image = resized_image.convert("L")
            normalized_image = np.array(grayscale_image) / 255.0

            # Reshape the image to match the model's input shape
            input_image = normalized_image.reshape((1, 28, 28))

            # Make the prediction
            prediction = self.model.predict(input_image)
            predicted_digit = prediction.argmax()

            # Update the histogram and prediction label
            self.update_histogram(prediction[0])
            self.prediction_label.config(text=f"Prediction: {predicted_digit}")
        except Exception as e:
            print(f"Error updating prediction: {e}")

    def start(self):
        # Start the GUI main loop
        self.window.mainloop()


# Create an instance of HandwrittenDigitRecognition and start the application
app = HandwrittenDigitRecognition()
app.start()
