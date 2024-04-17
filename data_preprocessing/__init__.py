# import os
# from optparse import OptionParser
from .grid_extractor import GridExtractor
from .matrix_generator import MatrixGenerator

# def process_images_in_folder(folder_path):
#     """
#     Process all image files in the specified folder.

#     Args:
#         folder_path (str): The path to the folder containing image files.
#     """
#     files = os.listdir(folder_path)
#     image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     if not image_files:
#         print("No image files found in the folder.")
#         return
    
#     for image_file in image_files:
#         try:
#             image_path = os.path.join(folder_path, image_file)
#             print(f"Processing {image_path}")
#             grid_converter = GridExtractor(image_path)
#             grid_converter.preprocess_image()
#             extracted_img = grid_converter.extract_grid()
#             grid_converter.save_grid(extracted_img)
#             matrix_generator = MatrixGenerator(image_path)
#             matrix_generator.save_matrix(extracted_img)
#         except Exception:
#             print("Error faced: ", Exception)

# if __name__ == "__main__":
#     parser = OptionParser()
#     parser.add_option("-f", "--folder", dest="folder_path", help="Path to the folder containing image files")
#     (options, args) = parser.parse_args()

#     if not options.folder_path:
#         parser.error("Path to the folder containing image files is required. Use -f or --folder option.")

#     process_images_in_folder(options.folder_path)
