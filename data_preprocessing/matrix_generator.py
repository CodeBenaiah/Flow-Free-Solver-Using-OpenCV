import cv2 as cv
import json
import numpy as np
import math
import os

class MatrixGenerator:
    """Class for generating a matrix from an image."""

    def __init__(self, image_path) -> None:
        """Initialize the MatrixGenerator."""
        self.colors_dataset = self.load_color_dataset()
        self.image_path = image_path

    def load_color_dataset(self) -> dict:
        """Load the color dataset from a JSON file.

        Returns:
            dict: A dictionary containing color data.
        """
        color_dataset_path = "/home/ben/Code/Flow-Free-Solver/dataset/colors/dataset.json"
        try:
            with open(color_dataset_path, 'r') as file:
                colors_dataset = json.load(file)
            return colors_dataset
        except FileNotFoundError:
            print(f"Error: Color dataset file not found at {color_dataset_path}.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in color dataset file at {color_dataset_path}.")
            return {}

    def size_calculator(self, image) -> tuple:
        """Calculate the size of the matrix based on the image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            tuple: A tuple containing the number of rows and columns in the matrix.
        """
        edges = cv.Canny(image, 50, 150, apertureSize=3)
        lines = cv.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            print("Error detecting lines. Model Failed")
            return 0, 0

        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            if np.isclose(theta, 0) or np.isclose(theta, np.pi):
                if not any(abs(x0 - x) < 10 for x in vertical_lines):
                    vertical_lines.append(x0)
            elif np.isclose(theta, np.pi / 2) or np.isclose(theta, 3 * np.pi / 2):
                if not any(abs(y0 - y) < 10 for y in horizontal_lines):
                    horizontal_lines.append(y0)

        num_rows = len(vertical_lines) - 1
        num_cols = len(horizontal_lines) - 1

        return num_rows, num_cols

    def split_boxes(self, image, n,m) -> list:
        """Split the image into boxes.

        Args:
            image (numpy.ndarray): The input image.
            n (int): The size of the matrix.

        Returns:
            list: List of boxes containing parts of the image.
        """
        rows = np.array_split(image, n, axis=0)
        boxes = []
        for r in rows:
            cols = np.array_split(r, m, axis=1)
            for box in cols:
                boxes.append(box)
        return boxes

    def predict(self, boxes) -> list:
        """Predict the values for each box.

        Args:
            boxes (list): List of boxes containing parts of the image.

        Returns:
            list: List of predicted values for each box.
        """
        board = []
        for image in boxes:
            color = image[50, 50]  # Get color at the center of the box
            print(color)
            matched_color = self.match_color(color)
            if matched_color is not None:
                board.append(matched_color)
        return board

    def match_color(self, color) -> int:
        """Match the color from the image with colors in the dataset.

        Args:
            color (tuple): Color information.

        Returns:
            int: The matched color value.
        """
        for key, val in self.colors_dataset.items():
            if np.array_equal(color, val):
                return int(key)
        return 0

    def save_matrix(self, image) -> None:
        """Save the generated matrix to a text file.

        Args:
            image (numpy.ndarray): The input image.
        """
        n,m = self.size_calculator(image)
        boxes = self.split_boxes(image, n,m)
        board = self.predict(boxes)
        file_name = os.path.basename(self.image_path)
        name, _ = os.path.splitext(file_name)
        output_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_matrices", "txt", name+"txt")
        with open(output_path, 'w') as file:
            file.write(str(n) + "\n")
            file.write(" ".join(str(cell) for cell in board))
            print(f"Matrix Extracted and successfully saved at {name}.")
            file.close()
