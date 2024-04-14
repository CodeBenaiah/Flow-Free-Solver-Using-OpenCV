import cv2
import numpy as np
import os
from typing import Optional, Tuple

class GridExtractor:
    """
    Class to extract the grid from an input image.

    Attributes:
        width (int): The width of the output grid image.
        height (int): The height of the output grid image.
        image_path (str): The path to the input image file.
        raw_image (numpy.ndarray): The raw input image.
    """

    def __init__(self, image_path: str, width: int = 500, height: int = 500) -> None:
        """
        Initialize the GridExtractor with the path to the input image and output dimensions.

        Args:
            image_path (str): The path to the input image file.
            width (int, optional): The width of the output grid image. Default is 500.
            height (int, optional): The height of the output grid image. Default is 500.
        """
        self.width = width
        self.height = height
        self.image_path = image_path
        self.raw_image = cv2.imread(image_path)
        self.preprocessed_image = self._preprocess_image()

    def _preprocess_image(self) -> np.ndarray:
        """
        Preprocess the input image.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
        canny_image = cv2.Canny(self.raw_image, 100, 200)
        return canny_image

    def _reorder(self, points: np.ndarray) -> np.ndarray:
        """
        Reorder the points in the contour to a specific order.

        Args:
            points (numpy.ndarray): The points in the contour.

        Returns:
            numpy.ndarray: The reordered points.
        """
        points = points.reshape((4, 2))
        points_new = np.zeros((4, 1, 2), dtype=np.int32)
        add = points.sum(1)
        points_new[0] = points[np.argmin(add)]
        points_new[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        points_new[1] = points[np.argmin(diff)]
        points_new[2] = points[np.argmax(diff)]
        return points_new

    def _find_largest_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest contour in the given image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            Optional[numpy.ndarray]: The largest contour found, or None if no suitable contour is found.
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 60000:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.002 * perimeter, True)
                if area > max_area and len(approx) == 4:
                    max_area = area
                    largest_contour = approx

        return largest_contour

    def extract_grid(self) -> Optional[np.ndarray]:
        """
        Extract the grid from the preprocessed image.

        Returns:
            Optional[numpy.ndarray]: The extracted grid image, or None if no grid is detected.
        """
        largest_contour = self._find_largest_contour(self.preprocessed_image)
        if largest_contour is not None:
            largest_contour = self._reorder(largest_contour)
            pts1 = np.float32(largest_contour)
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped_image = cv2.warpPerspective(self.raw_image, matrix, (self.width, self.height))
            return warped_image
        else:
            print("No grid detected in the image.")
            return None

    def save_grid(self, grid_image: Optional[np.ndarray]) -> None:
        """
        Save the extracted grid image to the raw_matrices folder with the same name as the input image.

        Args:
            grid_image (Optional[numpy.ndarray]): The grid image to be saved.
        """
        if grid_image is not None:
            file_name = os.path.basename(self.image_path)
            output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_matrices", "Image", file_name)
            cv2.imwrite(output_dir, grid_image)
            print(f"Grid extracted and saved successfully as {file_name}.")
        else:
            print("Failed to save the grid image. No grid detected or extracted.")