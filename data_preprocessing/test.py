from matrix_generator import MatrixGenerator
import cv2 as cv
import math
import numpy as np

if __name__ == "__main__":
    image_path = "/home/ben/Code/Flow-Free-Solver/dataset/raw_matrices/Image/Screenshot_20240219-155921.png"
    matrix_generator = MatrixGenerator(image_path)
    image = cv.imread(image_path)

                                                                    # Apply Canny edge detection
    edges = cv.Canny(image, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        print("Error detecting lines. Model Failed")

    # Initialize lists to store vertical and horizontal lines
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        rho, theta = line[0]

        # Compute line coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Filter vertical and horizontal lines based on their angles
        if np.isclose(theta, 0) or np.isclose(theta, np.pi):
            if not any(abs(x0 - x) < 10 for x in vertical_lines):
                vertical_lines.append(x0)
        elif np.isclose(theta, np.pi / 2) or np.isclose(theta, 3 * np.pi / 2):
            if not any(abs(y0 - y) < 10 for y in horizontal_lines):
                horizontal_lines.append(y0)

    # Calculate number of rows and columns
    num_rows = len(vertical_lines) - 1
    num_cols = len(horizontal_lines) - 1                    
        
    print("Number of vertical lines:", num_rows,num_cols)
    
    cv.waitKey()
    cv.destroyAllWindows()
