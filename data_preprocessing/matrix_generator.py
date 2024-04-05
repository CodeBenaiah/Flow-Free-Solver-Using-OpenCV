import cv2 as cv
import json
import numpy as np
import math
import os
from collections import defaultdict

class MatrixGenerator:
    """Class for generating a matrix from an image."""

    def __init__(self, image_path) -> None:
        """Initialize the MatrixGenerator."""
        self.colors_dataset = defaultdict(list)
        self.image_path = image_path

    def load_color_dataset(self, colors) -> dict:
        """Load the color dataset from a JSON file.

        Returns:
            dict: A dictionary containing color data.
        """
        while colors:
            temp = colors[0]
            min_distance = 99999
            min_color = None
            for color in colors[1:]:
                temp_distance = [abs(c1 - c2) for c1, c2 in zip(temp, color)]
                distance = sum(temp_distance)
                if distance < min_distance:
                    min_distance = distance
                    min_color = color

            self.colors_dataset[len(self.colors_dataset) + 1] = [temp, min_color]
            colors.remove(temp)
            colors.remove(min_color)

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
        color_list = []

        for r in rows:
            cols = np.array_split(r, m, axis=1)
            for box in cols:
                b, g, r = cv.split(box)
                max_r = np.max(r)
                max_g = np.max(g)
                max_b = np.max(b)
                if not(max_r < 125 and max_g < 125 and max_b < 125):
                    color_list.append([max_r, max_g, max_b])
                boxes.append(box)

        return boxes, color_list

    def predict(self, boxes) -> list:
        """Predict the values for each box.

        Args:
            boxes (list): List of boxes containing parts of the image.

        Returns:
            list: List of predicted values for each box.
        """
        board = []
        for image in boxes:
            b, g, r = cv.split(image)
            max_r = np.max(r)
            max_g = np.max(g)
            max_b = np.max(b)
            if max_r < 150 and max_g < 150 and max_b < 150:
                board.append(0)
            else:
                matched_color = self.match_color([max_r, max_g, max_b])
                board.append(matched_color)
                
        return board

    def match_color(self, color) -> int:
        """Match the color from the image with colors in the dataset.

        Args:
            color (tuple): Color information.

        Returns:
            int: The matched color value.
        """
        for key,val in self.colors_dataset.items():
            if color in val:
                return key
        
        return 0

    def save_matrix(self, image) -> None:
        """Save the generated matrix to a text file.

        Args:
            image (numpy.ndarray): The input image.
        """
        n,m = self.size_calculator(image)
        boxes, color_list = self.split_boxes(image, n,m)
        self.load_color_dataset(color_list)
        board = self.predict(boxes)
        file_name = os.path.basename(self.image_path)
        name, _ = os.path.splitext(file_name)
        output_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_matrices", "txt", name+"txt")
        with open(output_path, 'w') as file:
            file.write(str(n) + "\n")
            file.write(" ".join(str(cell) for cell in board))
            print(f"Matrix Extracted and successfully saved at {name}.")
            file.close()
