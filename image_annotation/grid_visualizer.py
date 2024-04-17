import os
from typing import List, Tuple

import cv2
import numpy


class Visualizer:
    """
    A class for visualizing the solution of a grid-based problem on an image.

    Attributes:
        color_dataset (List[Tuple[int, int, int]]): A list of RGB color tuples representing the color palette.
        grid_image (numpy.ndarray): The input image on which the solution will be visualized.
        solution (List[int]): The solution to the grid-based problem.
        row_size (int): The number of rows in the grid.
        col_size (int): The number of columns in the grid.
    """

    def __init__(
        self,
        color_dataset: List[Tuple[int, int, int]],
        image: numpy.ndarray,
        row_size: int,
        col_size: int,
        image_path: str,
    ) -> None:
        """
        Initialize the Visualizer instance.

        Args:
            color_dataset (List[Tuple[int, int, int]]): A list of RGB color tuples representing the color palette.
            image (numpy.ndarray): The input image on which the solution will be visualized.
            row_size (int): The number of rows in the grid.
            col_size (int): The number of columns in the grid.
            image_path (str): The path to the input image file.
        """
        self.color_dataset = color_dataset
        self.grid_image = image
        self.solution = self._read_solution()
        self.row_size = row_size
        self.col_size = col_size
        self.image_path = image_path
        for key, val in self.color_dataset.items():
            print(key, val[0])

    def _read_solution(self) -> List[int]:
        """
        Read the solution from a text file.

        Returns:
            List[int]: A list of integers representing the solution.
        """
        with open("dataset/solution_matrices/output.txt", "r") as solution_file:
            data = solution_file.read()
            return [int(x) for x in data.split()]

    def display_output(self) -> None:
        """
        Visualize the solution on the input image by drawing rectangles with colors from the color palette.
        """
        rect_width = self.grid_image.shape[1] // self.col_size
        rect_height = self.grid_image.shape[0] // self.row_size
        solution_idx = 0
        print(self.solution)
        for row in range(self.row_size):
            row_start = row * rect_height
            row_end = row_start + rect_height

            for col in range(self.col_size):
                col_start = col * rect_width
                col_end = col_start + rect_width
                b, g, r = map(int, self.color_dataset[self.solution[solution_idx]][0])
                cv2.rectangle(
                    self.grid_image,
                    pt1=(col_start, row_start),
                    pt2=(col_end, row_end),
                    color=(b, g, r),
                    thickness=-1,
                )
                solution_idx += 1

    def save_image(self, output_dir: str) -> None:
        """
        Save the annotated image to the specified output directory.

        Args:
            output_dir (str): The path to the output directory where the annotated image will be saved.
        """
        if self.grid_image is not None:
            file_name = os.path.basename(self.image_path)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, self.grid_image)
            print(f"Grid annotated and saved as {file_name}.")
        else:
            print("Failed to annotate the image, incorrect solution found.")
