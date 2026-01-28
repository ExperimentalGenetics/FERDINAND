import os

import cv2
import numpy as np
import pandas as pd

from ferdinand.image_procs import resize_image_to_array
from ferdinand.utils import build_local_image_dir

import ferdinand.sqlite_procs as sqlp
import ferdinand.image_procs as imgp
import ferdinand.utils as utl
import ferdinand.preprocess_and_qc as ppqc

def preprocess_image_from_BCM(image):
    """
    Preprocess the given image from BCM using a series of image enhancement techniques.
    :param image: The image to be processed.
    :return: tuple
        - result_image (ndarray): The processed image after applying the transformations.
        - methods_used (list of str): A list of method names that were applied to the image
                                      during pre-processing, in the order they were applied.
    """

    methods_used = []

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    preproc_img, method_name = imgp.crop_image_from_right(preproc_img.copy())
    methods_used.append(method_name)

    is_inverted = imgp.detect_image_inversion(preproc_img)
    if is_inverted:
        preproc_img, method_name = imgp.invert_image(preproc_img.copy())
        methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def preprocess_image_from_HMGU(image):
    """
    @TODO
    :param image:
    :return:
    """
    methods_used = []
    # kernel_setting_value = 1

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    #is_inverted = imgproc.detect_image_inversion(preproc_img2)
    #if is_inverted:
    #    preproc_img3, method_name = imgproc.apply_morphological_operations(preproc_img2, kernel_size=kernel_setting_value)
    #    methods_used.append(method_name)
    #else:
    #    preproc_img3 = preproc_img2

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def preprocess_image_from_ICS(image):
    """
    @TODO
    Preprocess the input image from mrc ics using a series of image enhancement techniques.

    Args: image (ndarray): The input image to be processed. This should be compatible with
                         the `image_utils` library used for transformations.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """
    methods_used = []

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    is_inverted = imgp.detect_image_inversion(preproc_img.copy())
    if is_inverted:
        preproc_img, method_name = imgp.crop_image_from_top(preproc_img.copy())
        methods_used.append(method_name)
        preproc_img, method_name = imgp.invert_image(preproc_img.copy())
        methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def _standard_preprocess_image_from_center(image):
    """
    Apply a standard series of image enhancement techniques to preprocess the input image.
    Args:
        image (ndarray): The input image to be processed.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """

    methods_used = []

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def preprocess_image_from_JAX(image):
    """
    @TODO
    Preprocess the input image from jax center using a series of image enhancement techniques.

    Args: image (ndarray): The input image to be processed. This should be compatible with
                         the `image_utils` library used for transformations.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """
    return _standard_preprocess_image_from_center(image)

def preprocess_image_from_KMPC(image):
    """
    @TODO
    Preprocess the input image from mrc kmpc using a series of image enhancement techniques.

    Args: image (ndarray): The input image to be processed. This should be compatible with
                         the `image_utils` library used for transformations.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """
    return _standard_preprocess_image_from_center(image)

def preprocess_image_from_MRC_Harwell(image):
    """
    @TODO
    Preprocess the input image from mrc harwell center using a series of image enhancement techniques.

    Args: image (ndarray): The input image to be processed. This should be compatible with
                         the `image_utils` library used for transformations.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """
    methods_used = []

    average_brightness, median_brightness = imgp.analyze_center_brightness(image)

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    if average_brightness > 200:
        threshold=100
        preproc_img, method_name = imgp.remove_marker(preproc_img.copy(), threshold=threshold)
        methods_used.append(method_name)
    elif 200 > average_brightness > 120:
        threshold=60
        preproc_img, method_name = imgp.remove_marker(preproc_img.copy(), threshold=threshold)
        methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def preprocess_image_from_RBRC(image):
    """
    @TODO
    :param image:
    :return:
    """
    return _standard_preprocess_image_from_center(image)

def preprocess_image_from_TCP(image):
    """
    Preprocess the input image from mrc tcp using a series of image enhancement techniques.

    Args: image (ndarray): The input image to be processed. This should be compatible with
                         the `image_utils` library used for transformations.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """
    methods_used = []

    average_brightness, median_brightness = imgp.analyze_center_brightness(image)

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    if average_brightness > 220:
        kernel_size = 1

        #preproc_img3, method_name=imgproc.apply_morphological_operations(preproc_img2, kernel_size=kernel_size)
        #methods_used.append(method_name)

        preproc_img, method_name = imgp.crop_image_from_right(preproc_img.copy())
        methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def preprocess_image_from_UC_Davis(image):
    """
    @TODO
    Preprocess the input image using a series of image enhancement techniques.
    Args:
        image (ndarray): The input image to be processed. This should be compatible with
                         the `image_utils` library used for transformations.
    Returns:
        tuple:
            result_image (ndarray): The processed image after applying the transformations.
            methods_used (list of str): A list of method names that were applied to the image
                                        during preprocessing, in the order they were applied.
    """
    methods_used = []

    average_brightness, median_brightness = imgp.analyze_center_brightness(image)

    preproc_img, method_name = imgp.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    if average_brightness < 150:
        preproc_img, method_name = imgp.apply_brightness(preproc_img.copy(), target_brightness=80)
        methods_used.append(method_name)
    #elif average_brightness > 200:
    #    kernel_setting_value = 1
    #    preproc_img3, method_name = imgproc.apply_morphological_operations(preproc_img2, kernel_setting_value)
    #    methods_used.append(method_name)

    preproc_img, method_name = imgp.apply_clahe(preproc_img.copy())
    methods_used.append(method_name)

    return preproc_img, methods_used

def preprocess_image_from_WTSI(image):
    """
    @TODO
    :param image:
    :return:
    """
    return _standard_preprocess_image_from_center(image)

# Map centers to their respective preprocessing functions
FUNC_MAP = {
    "BCM": preprocess_image_from_BCM,
    "HMGU": preprocess_image_from_HMGU,
    "ICS": preprocess_image_from_ICS,
    "JAX": preprocess_image_from_JAX,
    "KMPC": preprocess_image_from_KMPC,
    "MRC Harwell": preprocess_image_from_MRC_Harwell,
    "RBRC": preprocess_image_from_RBRC,
    "TCP": preprocess_image_from_TCP,
    "UC Davis": preprocess_image_from_UC_Davis,
    "WTSI": preprocess_image_from_WTSI
}

def preprocess_image(conn, db_table, center, image_file_path, source_path, target_path, 
                     pad_to_square=True, logger=None):
    """
    Preprocess the given image file based on the specified center's preprocessing function.
    :param conn: Active sqlite connection object.
    :param center: Name of centre where the image has been generated.
    :param image_file_path: Full path to the image file to be processed.
    :param source_path: Path to locally saved images.
    :param target_path: Path to where the processed images are locally stored.
    :param pad_to_square: Boolean indicating whether to pad the images to a square shape before saving (default: True).
    :param logger: Logger object for logging messages (default: None).
    :return: tuple (preproc_file_path, preproc_image, methods_used)"""

    preproc_func = FUNC_MAP.get(center)
    if preproc_func is None:
        raise ValueError(f"Unsupported center: {center}")

    # read the image
    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"ERROR: Image not found at {image_file_path}!")

    if source_path in os.path.dirname(image_file_path) and target_path is not None:
        os.makedirs(target_path, exist_ok=True)
        preproc_file_path = image_file_path.replace(source_path, target_path)
        os.makedirs(os.path.dirname(preproc_file_path), exist_ok=True)
    else: 
        raise ValueError(f"ERROR: image_path {source_path} not found in image_file_path {image_file_path} or {target_path} is not valid!")
    
    # perform pre-processing
    preproc_image, methods_used = preproc_func(image)
    if preproc_image is None:
        raise ValueError("Invalid image data provided. The image cannot be None or empty.")

    success = cv2.imwrite(str(preproc_file_path), preproc_image)
    if not success:
        raise IOError(f"Failed to write image: {preproc_file_path}")

    if pad_to_square:
        # pad to square, if defined
        preproc_image = imgp.pad_image_to_square(preproc_image.copy())
    
    # save the pre-processed image
    if preproc_image is None:
        raise ValueError("Invalid image data provided. The image cannot be None or empty.")
    success = cv2.imwrite(str(preproc_file_path), preproc_image)
    if not success:
        raise IOError(f"Failed to write image: {preproc_file_path}")
    else:
        if logger is not None:
            logger.debug(f"pre-processed image saved to: {preproc_file_path}")
        if conn is not None:
            sqlp.update_preprocess_status(conn=conn, 
                                         db_table=db_table, 
                                         status='yes', 
                                         omero_id=os.path.splitext(os.path.basename(image_file_path))[0], 
                                         preproc_methods=methods_used, 
                                         img_height=preproc_image.shape[0],
                                         img_width=preproc_image.shape[1],
                                         logger=logger)
    return preproc_file_path, preproc_image, methods_used    

def preprocess_images(conn, db_table, center, image_files, source_path, target_path,
                      pad_to_square=True, logger=None, no_of_images_to_show=5):
    """
    Processes and prepares a list of images for a specific center, applies necessary transformations, and stores them in the archive.
    :param conn: Active sqlite connection object.
    :param center: Name of centre where the images have been generated.
    :param image_files: List of image file paths to be processed.
    :param source_path: Path to locally saved images.
    :param target_path: Path to where the processed images are locally stored.
    :param pad_to_square: Boolean indicating whether to pad the images to a square shape before saving (default: True).
    :param logger: Logger object for logging messages (default: None).
    :return: List of file paths to the pre-processed images.
    """

    preproc_img_files = []
    for img_file in image_files:
        try: 
            preproc_img_file, preproc_img, preproc_methods = preprocess_image(conn=conn, 
                                                                              db_table=db_table,
                                                                              center=center, 
                                                                              image_file_path=img_file, 
                                                                              source_path=source_path,
                                                                              target_path=target_path, 
                                                                              pad_to_square=pad_to_square, 
                                                                              logger=logger)
            preproc_img_files.append(preproc_img_file)
        except (FileNotFoundError, ValueError, IOError) as e: 
            logger.error(f"could not preprocess image {img_file}: {e}")
            continue
    if no_of_images_to_show > 0 and len(preproc_img_files) > 0:  
        utl.plot_image_grid(image_files=preproc_img_files[:no_of_images_to_show], cols=5)

    return preproc_img_files