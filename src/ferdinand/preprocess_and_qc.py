import os

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

from ferdinand.utils import build_local_image_dir

import ferdinand.sqlite_utils as sqlutl
import ferdinand.image_utils as imgutl
import ferdinand.utils as utl
import ferdinand.preprocess_and_qc as ppqc

"""
Preprocessing and quality-control helpers for IMPC image pipelines.

The module defines center-specific preprocessing routines, batch helpers that
write processed or rotated images to disk, and QC utilities that update SQLite
metadata based on exposure and orientation analysis.
"""

def preprocess_image_from_BCM(image):
    """
    Preprocess a BCM image with the BCM-specific enhancement sequence.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Preprocess an HMGU image with the HMGU-specific enhancement sequence.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Preprocess an ICS image with the ICS-specific enhancement sequence.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Apply the default preprocessing pipeline used by several centers.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Preprocess a JAX image with the default preprocessing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
    """
    return _standard_preprocess_image_from_center(image)

def preprocess_image_from_KMPC(image):
    """
    Preprocess a KMPC image with the default preprocessing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
    """
    return _standard_preprocess_image_from_center(image)

def preprocess_image_from_MRC_Harwell(image):
    """
    Preprocess an MRC Harwell image with brightness-dependent marker removal.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Preprocess an RBRC image with the default preprocessing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
    """
    return _standard_preprocess_image_from_center(image)

def preprocess_image_from_TCP(image):
    """
    Preprocess a TCP image with brightness-dependent right-edge cropping.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Preprocess a UC Davis image with brightness compensation when needed.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
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
    Preprocess a WTSI image with the default preprocessing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Processed image and the ordered list of preprocessing method names that
        were applied.
    """
    return _standard_preprocess_image_from_center(image)

# map centers to their respective preprocessing functions
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
    Preprocess one image, save it, and optionally update the SQLite metadata.

    Parameters
    ----------
    conn : sqlite3.Connection | None
        Open SQLite connection used to record preprocessing status.
    db_table : str
        Database table updated when `conn` is provided.
    center : str
        Center name used to select the preprocessing function from `FUNC_MAP`.
    image_file_path : str | os.PathLike
        Full path to the source image file.
    source_path : str | os.PathLike
        Root directory segment that should be replaced when constructing the
        output path.
    target_path : str | os.PathLike
        Root directory where the preprocessed image should be written.
    pad_to_square : bool, optional
        Whether to center the processed image onto a square canvas before the
        final save.
    logger : logging.Logger, optional
        Logger used for debug output.

    Returns
    -------
    tuple[str, numpy.ndarray, list[str]]
        Output file path, final preprocessed image, and the ordered list of
        preprocessing method names that were applied.

    Raises
    ------
    ValueError
        Raised when the center is unsupported, path rewriting is invalid, or
        preprocessing returns no image data.
    FileNotFoundError
        Raised when the input image cannot be read.
    IOError
        Raised when the processed image cannot be written to disk.
    """

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
        if center in ['BCM', 'MRC Harwell', 'ICS', 'RBRC', 'TCP']:
            center_mode = "bbox"
        else:
            center_mode = "centroid"
        # pad to square, if defined
        preproc_image = imgutl._center_mouse_on_square(
            preproc_image.copy(),
            use_replicate=False,
            center_mode=center_mode,
            output_mode="fixed_then_crop",
        )
        # preproc_image = imgutl.center_mouse_fixed_square(preproc_image.copy())
        # preproc_image = imgutl.pad_image_to_square(preproc_image.copy()) 
        
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
    Preprocess a batch of images for one center.

    Parameters
    ----------
    conn : sqlite3.Connection | None
        Open SQLite connection used to record preprocessing status.
    db_table : str
        Database table updated when `conn` is provided.
    center : str
        Center name used to select the preprocessing pipeline.
    image_files : sequence[str | os.PathLike]
        Source image files to preprocess.
    source_path : str | os.PathLike
        Root directory segment that should be replaced when constructing output
        paths.
    target_path : str | os.PathLike
        Root directory where preprocessed images should be written.
    pad_to_square : bool, optional
        Whether to center processed images on a square canvas.
    logger : logging.Logger, optional
        Logger used for per-image error messages.
    no_of_images_to_show : int, optional
        Number of processed images to preview with `plot_image_grid`.

    Returns
    -------
    list[str]
        File paths to successfully preprocessed images.
    """
    preproc_img_files = []
    pbar = tqdm(image_files, desc="Preprocessing images")
    
    for img_file in pbar:
        pbar.set_postfix_str(str(img_file))
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
    print()
    if no_of_images_to_show > 0 and len(preproc_img_files) > 0:  
        utl.plot_image_grid(image_files=preproc_img_files[:no_of_images_to_show], cols=5)

    return preproc_img_files

def flag_overexposed_images(conn, db_table, rows: pd.DataFrame, source_path, logger=None): 
    """
    Flag overexposed images in the database using brightness heuristics.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection used for schema updates and flag writes.
    db_table : str
        Database table to update.
    rows : pandas.DataFrame
        Rows containing at least `center`, `cohort_type`, `gene_symbol`,
        `sex`, and `omero_id`.
    source_path : str | os.PathLike
        Root directory used to build the local JPEG path for each image.
    logger : logging.Logger, optional
        Logger forwarded to SQLite helper functions.

    Returns
    -------
    None
        The function updates database columns in place and does not return a
        value.
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
        
            is_object_too_bright = imgutl.analyze_mouse_brightness_median(image)
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
                
def predict_orientation_angle(conn, db_table, model, image_file_path, db_column_name, 
                              target_size=(224, 224),
                              preproc_func=None, image=None):
    """
    Predict an image orientation angle and optionally store it in SQLite.

    Parameters
    ----------
    conn : sqlite3.Connection | None
        Open SQLite connection used to retrieve metadata and optionally store
        the prediction.
    db_table : str
        Database table queried and updated when `conn` is provided.
    model : object
        Loaded model exposing a `predict(...)` method compatible with the image
        batch produced here.
    image_file_path : str | os.PathLike
        Full path to the image file.
    db_column_name : str | None
        Database column that should receive the predicted angle. When `None`,
        no prediction value is written back.
    target_size : tuple[int, int], optional
        Resize target passed to `imgutl.get_image_as_array`.
    preproc_func : callable | None, optional
        Optional preprocessing function applied to the resized batch before
        inference.
    image : numpy.ndarray | None, optional
        Optional preloaded grayscale image used only for existence validation.

    Returns
    -------
    int
        Predicted orientation angle as the winning class index.

    Raises
    ------
    FileNotFoundError
        Raised when the source image cannot be read.
    ValueError
        Raised when the database does not contain exactly one row for the image
        OMERO ID.
    """
    # load the image in grayscale
    if image is None:
        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_file_path}")

    # resize the image for the model
    resized_image_as_array = imgutl.get_image_as_array(image_file_path=image_file_path, 
                                                          target_size=target_size)

    if preproc_func is not None:
        resized_image_as_array = preproc_func(resized_image_as_array)

    predict = model.predict(resized_image_as_array)
    predicted_angle = np.argmax(predict, axis=1).item()

    # extract Omero ID from the file name
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

        # extract mouse ID from the database sample
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
    Convert a predicted orientation into the corrective rotation angle.

    Parameters
    ----------
    predicted_orientation_angle : int | float
        Predicted orientation angle in degrees.
    logger : logging.Logger, optional
        Logger used for debug output.

    Returns
    -------
    int | float
        Rotation angle in degrees that would bring the image back to the
        canonical orientation.
    """
    predicted_rotation_angle = (-predicted_orientation_angle)%360

    if logger is not None:
        logger.debug(f"Predicted orientation angle: {predicted_orientation_angle}")
        logger.debug(f"Predicted rotation angle: {predicted_rotation_angle}")   

    return predicted_rotation_angle

def load_and_rotate_image(conn, db_table, image_file, source_path, target_path, logger=None):
    """
    Rotate one image using a previously stored orientation prediction.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection used to read the stored orientation prediction
        and update rotation status.
    db_table : str
        Database table queried and updated during rotation.
    image_file : str | os.PathLike
        Full path to the source image file.
    source_path : str | os.PathLike
        Root directory segment that should be replaced when constructing the
        output path.
    target_path : str | os.PathLike
        Root directory where the rotated image should be written.
    logger : logging.Logger, optional
        Logger used for debug output.

    Returns
    -------
    tuple[str, numpy.ndarray, int | float]
        Output file path, rotated image, and corrective rotation angle.

    Raises
    ------
    FileNotFoundError
        Raised when the source image cannot be read.
    ValueError
        Raised when no stored orientation exists or output path rewriting is
        invalid.
    IOError
        Raised when the rotated image cannot be written to disk.
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
    Rotate a batch of images using previously stored orientation predictions.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection used to read predictions and update rotation
        status.
    db_table : str
        Database table queried and updated during rotation.
    image_files : sequence[str | os.PathLike]
        Source image files to rotate.
    source_path : str | os.PathLike
        Root directory segment that should be replaced when constructing output
        paths.
    target_path : str | os.PathLike
        Root directory where rotated images should be written.
    logger : logging.Logger, optional
        Logger used for per-image error messages.
    no_of_images_to_show : int, optional
        Number of rotated images to preview with `plot_image_grid`.

    Returns
    -------
    list[str]
        File paths to successfully rotated images.
    """

    rotated_img_files = []
    pbar = tqdm(image_files, desc="Rotating images")
    for img_file in pbar:
        pbar.set_postfix_str(f"{img_file}")
    # for img_file in image_files:
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
