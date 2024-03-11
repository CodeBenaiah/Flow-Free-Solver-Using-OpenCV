import cv2 as cv
import numpy as np
import os

class GridExtractor:
    """Class to extract the grid from an input image.

    Attributes:
        width (int): The width of the output grid image.
        height (int): The height of the output grid image.
        image_path (str): The path to the input image file.
        raw_image (numpy.ndarray): The raw input image.
    """

    def __init__(self, image_path: str) -> None:
        """Initialize the GridExtractor with the path to the input image.

        Args:
            image_path (str): The path to the input image file.
        """
        self.width, self.height = 500, 500
        self.image_path = image_path
        self.raw_image = cv.imread(image_path)  
        self.preprocessed_image = self.preprocess_image()
    
    def preprocess_image(self) -> np.ndarray:
        """Preprocess the input image.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
        # img_grayscale = cv.cvtColor(self.raw_image, cv.COLOR_BGR2GRAY)
        # img_blur = cv.GaussianBlur(img_grayscale, (5,5),1)
        # thresholded_image = cv.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
        canny_image = cv.Canny(self.raw_image, 100, 200)
        return canny_image
    
    def reorder(self, points):
        points = points.reshape((4,2))
        pointsnew = np.zeros((4,1,2),dtype= np.int32)
        add = points.sum(1)
        pointsnew[0] = points[np.argmin(add)]
        pointsnew[3] = points[np.argmax(add)]
        diff = np.diff(points,axis=1)
        pointsnew[1] = points[np.argmin(diff)]
        pointsnew[2] = points[np.argmax(diff)]
        return pointsnew
    
    def find_largest_contour(self, image: np.ndarray) -> np.ndarray:
        """Find the largest contour in the given image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The largest contour found.
        """
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 60000:
                perimeter = cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, 0.002 * perimeter, True)
                if area > max_area and len(approx) == 4:
                    max_area = area
                    largest_contour = approx
        return largest_contour
    
    def extract_grid(self) -> np.ndarray:
        """Extract the grid from the preprocessed image.

        Returns:
            numpy.ndarray: The extracted grid image.
        """
        largest_contour = self.find_largest_contour(self.preprocessed_image)
        if largest_contour is not None:
            largest_contour = self.reorder(largest_contour)
            pts1 = np.float32(largest_contour)
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            warped_image = cv.warpPerspective(self.raw_image, matrix, (self.width, self.height))
            return warped_image
        else:
            print("No grid detected in the image.")
            return None
    
    def save_grid(self, grid_image: np.ndarray) -> None:
        """Save the extracted grid image to the raw_matrices folder with the same name as the input image.

        Args:
            grid_image (numpy.ndarray): The grid image to be saved.
        """
        if grid_image is not None:
            file_name = os.path.basename(self.image_path)
            output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_matrices", file_name)
            cv.imwrite(output_dir, grid_image)
            print(f"Grid extracted and saved successfully as {file_name}.")
        else:
            print("Failed to save the grid image. No grid detected or extracted.")
