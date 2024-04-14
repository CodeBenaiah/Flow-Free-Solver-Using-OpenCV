import cv2
import json
import numpy as np
import math
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

class MatrixGenerator:
    """
    Class for generating a matrix representation from an image.

    Attributes:
        colors_dataset (Dict[int, List[Tuple[int, int, int]]]): A dictionary mapping color indices to lists of RGB color tuples.
        image_path (str): The path to the input image.
    """

    def __init__(self, image_path: str) -> None:
        """
        Initialize the MatrixGenerator instance.

        Args:
            image_path (str): The path to the input image.
        """
        self.colors_dataset = defaultdict(list)
        self.image_path = image_path

    def load_color_dataset(self, colors: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Load the color dataset from a list of RGB color tuples.

        Args:
            colors (List[Tuple[int, int, int]]): A list of RGB color tuples.

        Returns:
            Dict[int, List[Tuple[int, int, int]]]: A dictionary mapping color indices to lists of RGB color tuples.
        """
        while colors:
            temp_color = colors[0]
            min_distance = float('inf')
            min_color = None
            for color in colors[1:]:
                temp_distance = sum(abs(c1 - c2) for c1, c2 in zip(temp_color, color))
                if temp_distance < min_distance:
                    min_distance = temp_distance
                    min_color = color

            color_index = len(self.colors_dataset) + 1
            self.colors_dataset[color_index] = [temp_color, min_color]
            colors.remove(temp_color)
            colors.remove(min_color)

        return self.colors_dataset

    def size_calculator(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Calculate the size of the matrix based on the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            Tuple[int, int]: A tuple containing the number of rows and columns in the matrix.
        """
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

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

    def split_boxes(self, image: np.ndarray, num_rows: int, num_cols: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
        """
        Split the input image into boxes and extract the dominant colors.

        Args:
            image (np.ndarray): The input image.
            num_rows (int): The number of rows in the matrix.
            num_cols (int): The number of columns in the matrix.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[int, int, int]]]: A tuple containing a list of boxes (numpy arrays) and a list of dominant RGB colors.
        """
        rows = np.array_split(image, num_rows, axis=0)
        boxes = []
        color_list = []

        for row in rows:
            cols = np.array_split(row, num_cols, axis=1)
            for box in cols:
                b, g, r = cv2.split(box)
                max_r = np.max(r)
                max_g = np.max(g)
                max_b = np.max(b)
                if not (max_r < 125 and max_g < 125 and max_b < 125):
                    color_list.append((max_r, max_g, max_b))
                boxes.append(box)

        return boxes, color_list

    def predict(self, boxes: List[np.ndarray]) -> List[int]:
        """
        Predict the values for each box based on the dominant color.

        Args:
            boxes (List[np.ndarray]): A list of boxes (numpy arrays) containing parts of the image.

        Returns:
            List[int]: A list of predicted values for each box.
        """
        board = []
        for image in boxes:
            b, g, r = cv2.split(image)
            max_r = np.max(r)
            max_g = np.max(g)
            max_b = np.max(b)
            if max_r < 150 and max_g < 150 and max_b < 150:
                board.append(0)
            else:
                matched_color = self.match_color((max_r, max_g, max_b))
                board.append(matched_color)

        return board

    def match_color(self, color: Tuple[int, int, int]) -> Optional[int]:
        """
        Match the color from the image with colors in the dataset.

        Args:
            color (Tuple[int, int, int]): An RGB color tuple.

        Returns:
            Optional[int]: The matched color index, or None if no match is found.
        """
        for color_index, color_list in self.colors_dataset.items():
            if color in color_list:
                return color_index

        return None

    def save_matrix(self, image: np.ndarray) -> Tuple[Dict[int, List[Tuple[int, int, int]]], int, int]:
        """
        Save the generated matrix to a text file.

        Args:
            image (np.ndarray): The input image.

        Returns:
            Tuple[Dict[int, List[Tuple[int, int, int]]], int, int]: A tuple containing the colors dataset, number of rows, and number of columns.
        """
        num_rows, num_cols = self.size_calculator(image)
        boxes, color_list = self.split_boxes(image, num_rows, num_cols)
        colors_dataset = self.load_color_dataset(color_list)
        board = self.predict(boxes)
        file_name = os.path.basename(self.image_path)
        name, _ = os.path.splitext(file_name)
        output_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_matrices", "txt", name + ".txt")
        with open(output_path, 'w') as file:
            file.write(f"{num_rows} {num_cols}\n")
            file.write(" ".join(str(cell) for cell in board))
            print(f"Matrix Extracted and successfully saved at {name}.txt")

        return colors_dataset, num_rows, num_cols