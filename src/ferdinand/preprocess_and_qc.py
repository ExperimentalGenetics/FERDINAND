import os

import cv2
import numpy as np
import pandas as pd

from ferdinand.utils import build_local_image_dir

import ferdinand.sqlite_utils as sqlutl
import ferdinand.image_utils as imgutl
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

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.crop_image_from_right(preproc_img.copy())
    methods_used.append(method_name)

    is_inverted = imgutl.detect_image_inversion(preproc_img)
    if is_inverted:
        preproc_img, method_name = imgutl.invert_image(preproc_img.copy())
        methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    is_inverted = imgutl.detect_image_inversion(preproc_img.copy())
    if is_inverted:
        preproc_img, method_name = imgutl.crop_image_from_top(preproc_img.copy())
        methods_used.append(method_name)
        preproc_img, method_name = imgutl.invert_image(preproc_img.copy())
        methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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

    average_brightness, median_brightness = imgutl.analyze_center_brightness(image)

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    if average_brightness > 200:
        threshold=100
        preproc_img, method_name = imgutl.remove_marker(preproc_img.copy(), threshold=threshold)
        methods_used.append(method_name)
    elif 200 > average_brightness > 120:
        threshold=60
        preproc_img, method_name = imgutl.remove_marker(preproc_img.copy(), threshold=threshold)
        methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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

    average_brightness, median_brightness = imgutl.analyze_center_brightness(image)

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    if average_brightness > 220:
        kernel_size = 1

        preproc_img, method_name = imgutl.crop_image_from_right(preproc_img.copy())
        methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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

    average_brightness, median_brightness = imgutl.analyze_center_brightness(image)

    preproc_img, method_name = imgutl.apply_gaussian_blur(image)
    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_edge_enhancement(preproc_img.copy())
    methods_used.append(method_name)

    if average_brightness < 150:
        preproc_img, method_name = imgutl.apply_brightness(preproc_img.copy(), target_brightness=80)
        methods_used.append(method_name)
    #elif average_brightness > 200:
    #    kernel_setting_value = 1
    #    preproc_img3, method_name = imguroc.apply_morphological_operations(preproc_img2, kernel_setting_value)
    #    methods_used.append(method_name)

    preproc_img, method_name = imgutl.apply_clahe(preproc_img.copy())
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
        preproc_image = imgutl.pad_image_to_square(preproc_image.copy())
    
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
            sqlutl.update_preprocess_status(conn=conn, 
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

def flag_overexposed_images(conn, db_table, rows: pd.DataFrame, source_path, logger=None): 

    """
    Flags images as overexposed based on brightness analysis and updates the database accordingly.
    
    :param conn: Active sqlite connection object.
    :param db_table: Name of the database table to update.
    :param rows: DataFrame containing rows for which overexposure is to be flagged.
    :type rows: pd.DataFrame
    :param source_path: Path to the local image directory.
    :param logger: Logger object for logging messages (default: None).
    """

    if not rows.empty:

        _df = rows.copy()

        for idx, row in _df.iterrows():
            image_file = os.path.join(utl.build_local_image_dir(source_path, row['center'], row['cohort_type'], row['gene_symbol'], row['sex']), 
                                      f"{row['omero_id']}.jpg")
        
            image = cv2.imread(image_file)
            if image is None: 
                print(f"Warning: Could not read image at location: {image_file}")
                continue
        
            is_object_too_bright = imgutl.analyze_maus_brightness_median(image)
            is_overall_too_bright= imgutl.detect_global_overexposure(image)

            for col, val in {'overexposed': is_object_too_bright, 
                             'overall_overexposed': is_overall_too_bright}.items():
                sqlutl.add_column_to_table(conn=conn, 
                                        db_table=db_table, 
                                        column_name=col,
                                        column_type='TEXT', 
                                        logger=logger)
                sqlutl.update_column_values(conn=conn, 
                                         db_table=db_table, 
                                         column=col, 
                                         value='yes' if val else 'no',
                                         condition_column='omero_id',
                                         condition_value=row['omero_id'])
                
def predict_orientation_angle(conn, db_table, model, image_file_path, db_column_name, target_size=(224, 224),
                              preproc_func=None, image=None):
    """
    Predicts the orientation angle of an image using a pre-trained model and updates the database with the predicted angle.
    
    :param conn: Active sqlite connection object.
    :param model: Loaded model used for angle prediction.
    :param image_file_path: Full path to the image file to be processed.
    :param db_column_name: Name of the database column to store the predicted angle.
    :param db_table: Name of the database table to update.
    :param preproc_func: Function for preprocessing the image before prediction (default: imgproc.binarize_images).
    :param image: Optional pre-loaded image data. If not provided, the image will be read from the file path.
    :return: Predicted angle of the image
    """

    # Load the image in grayscale
    if image is None:
        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_file_path}")

    # Resize the image for the model
    resized_image_as_array = imgutl.resize_image_to_array(image_file_path=image_file_path, 
                                                          img=image, target_size=target_size)

    if preproc_func is not None:
        resized_image_as_array = preproc_func(resized_image_as_array)

    predict = model.predict(np.array(resized_image_as_array))
    predicted_angle = np.argmax(predict, axis=1).item()

    # Extract Omero ID from the file name
    image_file_name = os.path.basename(image_file_path)
    omero_id, _ = os.path.splitext(image_file_name)

    mouse_id = None
    if conn is not None:
        # Retrieve image metadata from the SQLite database
        samples = sqlutl.select_rows_by_column(conn=conn, 
                                            db_table=db_table, 
                                            column='omero_id',
                                            value=int(omero_id))
        if len(samples) != 1:
            raise ValueError(f"Expected exactly one row in the database for image with Omero ID {omero_id}, "
                             f"but found {len(samples)}.")

        # Extract mouse ID from the database sample
        mouse_id = samples["mouse_id"].item()

        if db_column_name is not None:
            sqlutl.add_column_to_table(conn, db_table, db_column_name, 'INTEGER')
            rows_updated = sqlutl.update_column_values(conn=conn, 
                                                      db_table=db_table,
                                                      column=db_column_name,
                                                      value=int(predicted_angle),
                                                      condition_column='omero_id',
                                                      condition_value=int(omero_id))
            if rows_updated == 0:
                print("No rows updated — check omero_id/mouse_id")

    print(f"Omero ID: {omero_id}, Mouse ID: {mouse_id}, Predicted orientation angle: {predicted_angle}")

    return predicted_angle

def predict_rotation_angle(predicted_orientation_angle, logger=None):
    """
    Predicts the rotation angle needed to correct the image orientation based on the predicted orientation angle.
    
    :param predicted_orientation_angle: The predicted orientation angle of the image (in degrees).
    :param logger: Logger object for logging messages (default: None).
    
    :return: The angle difference (in degrees) needed to rotate the image to the correct orientation.
    """
    predicted_rotation_angle = (-predicted_orientation_angle)%360

    if logger is not None:
        logger.debug(f"Predicted orientation angle: {predicted_orientation_angle}")
        logger.debug(f"Predicted rotation angle: {predicted_rotation_angle}")   

    return predicted_rotation_angle

def load_and_rotate_image(conn, db_table, image_file, source_path, target_path, logger=None):
    """
    Loads an image, predicts its orientation angle using a pre-trained model, rotates the image accordingly, and saves the rotated image to the target path. 
    The function also updates the database with the rotation status and new image dimensions.
    
    :param conn: Active sqlite connection object.
    :param db_table: Name of the database table to update.
    :param image_file: Name of the image file to be processed.
    :param source_path: Path to the source directory where the image is located.

    :param target_path: Description
    :param logger: Description
    
    :return: tuple (rotated_file_path, rotated_image, rotation_angle)
        - rotated_file_path (str): The file path where the rotated image is saved.
        - rotated_image (ndarray): The rotated image after applying the rotation correction.
        - rotation_angle (float): The angle difference (in degrees) used to rotate the image
    """
    
    omero_id = os.path.splitext(os.path.basename(image_file))[0]

    # image_file_path = os.path.join(source_path, image_file)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"ERROR: Image not found at {image_file}!")


    orientation_angle = sqlutl.get_value_by_id(column='fst_angle_prediction', 
                                               omero_id=omero_id, 
                                               conn=conn, 
                                               db_table=db_table)
    if orientation_angle is None:
        raise ValueError(f"ERROR: No predicted angle found for image with Omero ID {omero_id}!")
    
    rotation_angle = predict_rotation_angle(orientation_angle, logger=logger)

    h, w = image.shape[:2]

    border_thickness = int(min(h, w) * 0.10)
    # print(border_thickness)
    top_strip = image[:border_thickness, :]
    bottom_strip = image[-border_thickness:, :]
    left_strip = image[:, :border_thickness]
    right_strip = image[:, -border_thickness:]

    border_pixels = np.concatenate([
        top_strip.reshape(-1, 2),
        bottom_strip.reshape(-1, 2),
        left_strip.reshape(-1, 2),
        right_strip.reshape(-1, 2)
    ], axis=0)

    mean_color = border_pixels.mean(axis=0).astype(np.uint8)

    rotated_image = imgutl.rotate_image(image, rotation_angle, background_color=mean_color.tolist())

    sqlutl.add_column_to_table(conn, db_table, 'rotated', 'TEXT', default_value='no')
    if rotated_image is not None:

        if source_path in os.path.dirname(image_file) and target_path is not None:
            os.makedirs(target_path, exist_ok=True)
            rotated_file_path = image_file.replace(source_path, target_path)
            os.makedirs(os.path.dirname(rotated_file_path), exist_ok=True)
        else: 
            raise ValueError(f"ERROR: image_path {source_path} not found in image_file {image_file} or {target_path} is not valid!")

        success = cv2.imwrite(str(rotated_file_path), rotated_image)
        if not success:
            raise IOError(f"Failed to write rotated image: {rotated_file_path}")
        else:
            if logger is not None:
                logger.debug(f"rotated image saved to: {rotated_file_path}")
            sqlutl.update_rotate_status(conn=conn, 
                                        db_table=db_table,
                                        status='yes', 
                                        omero_id=omero_id, 
                                        img_height=rotated_image.shape[0],
                                        img_width=rotated_image.shape[1],
                                        logger=logger)
              

    return rotated_file_path, rotated_image, rotation_angle

def load_and_rotate_images(conn, db_table, image_files, source_path, target_path, 
                           logger=None, no_of_images_to_show=5):
    """
    Processes and rotates a list of images based on predicted rotation angles, and stores them in the archive.
    
    :param conn: Active sqlite connection object.
    :param image_files: List of image file paths to be processed.
    :param source_path: Path to locally saved images.
    :param target_path: Path to where the processed images are locally stored.
    :param logger: Logger object for logging messages (default: None).
    
    :return: List of file paths to the rotated images.
    """

    rotated_img_files = []
    for img_file in image_files:
        try: 
            rotated_img_file, rotated_image, rotation_angle = load_and_rotate_image(conn=conn, 
                                                                                    db_table=db_table,
                                                                                    image_file=img_file, 
                                                                                    source_path=source_path,
                                                                                    target_path=target_path, 
                                                                                    logger=logger)
            rotated_img_files.append(rotated_img_file)
        except (FileNotFoundError, ValueError, IOError) as e: 
            logger.error(f"could not preprocess image {img_file}: {e}")
            continue
    if no_of_images_to_show > 0 and len(rotated_img_files) > 0:  
        utl.plot_image_grid(image_files=rotated_img_files[:no_of_images_to_show], cols=5)

    return rotated_img_files