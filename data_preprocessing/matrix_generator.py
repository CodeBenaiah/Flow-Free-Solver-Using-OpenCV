import cv2 as cv
import json
import numpy as np
from math import sqrt
import os

class MatrixGenerator:
    """Class for generating a matrix from an image."""

    def __init__(self) -> None:
        """Initialize the MatrixGenerator."""
        self.colors_dataset = self.load_color_dataset()

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

    def size_calculator(self, image) -> int:
        """Calculate the size of the matrix based on the image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            int: The size of the matrix.
        """
        imgGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        imgCanny = cv.Canny(image, 100, 200)

        contours, _ = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        min_area = float('inf')
        for cnt in contours:
            area = cv.contourArea(cnt)
            if 10 < area < min_area:
                min_area = area

        n = int(sqrt(250000 / min_area))
        return n
        
    def split_boxes(self, image, n) -> list:
        """Split the image into boxes.

        Args:
            image (numpy.ndarray): The input image.
            n (int): The size of the matrix.

        Returns:
            list: List of boxes containing parts of the image.
        """
        rows = np.vsplit(image, n)
        boxes = []
        for r in rows:
            cols = np.hsplit(r, n)
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
        return -1

    def save_matrix(self, image) -> None:
        """Save the generated matrix to a text file.

        Args:
            image (numpy.ndarray): The input image.
        """
        n = self.size_calculator(image)
        boxes = self.split_boxes(image, n)
        board = self.predict(boxes)
        name, _ = os.path.splitext(image)
        with open(f"{name}.txt", 'w') as file:
            file.write(str(n) + "\n")
            file.write(" ".join(str(cell) for cell in board))
