
import pydicom
import os
import cv2

import numpy as np
import PIL.ImageOps as ImageOps

from pydicom.errors import InvalidDicomError
from pydicom.misc import is_dicom
from PIL import Image
from io import BytesIO

def detect_file_format(response):
    """
    Detect the file format from a requests.Response object
    :param response: requests.Response object from requests.get()
    :return: Detected file format, e.g. "JPEG", "PNG", "DICOM", or "Unknown".
    """
    # 1. Check HTTP Content-Type header
    content_type = response.headers.get("Content-Type", "").lower()
    if "jpeg" in content_type:
        return "JPEG"
    elif "png" in content_type:
        return "PNG"
    elif "dicom" in content_type:
        return "DICOM"
    elif "bmp" in content_type:
        return "BMP"

    # 2. Try with PIL (JPEG, PNG, etc.)
    try:
        img = Image.open(BytesIO(response.content))
        return img.format  # e.g. "JPEG", "PNG"
    except Exception:
        pass

    # 3. Try with pydicom
    try:
        _ = pydicom.dcmread(BytesIO(response.content))
        return "DICOM"
    except Exception:
        pass

    return "Unknown"

def get_png_pixel_spacing(image, logger=None):

    """
    Extract pixel_spacing_x and pixel_spacing_y from a metadata dictionary.
    Tries 'dcm:PixelSpacing' first, then 'dcm:ImagerPixelSpacing'.
    Returns (pixel_spacing_x, pixel_spacing_y, manufacturer, manufacturer_model_name) or (None, None, None, None).
    """

    # validate input type
    if not isinstance(image, Image.Image):
        error_msg = f"Expected PIL Image object, got {type(image).__name__}"
        if logger:
            logger.error(error_msg)
        raise TypeError(error_msg)
    
    # extract manufacturer metadata if available
    manufacturer = image.info.get('dcm:Manufacturer')
    manufacturer_model_name = image.info.get("dcm:Manufacturer'sModelName")
    
    spacing_keys = ['dcm:PixelSpacing', 'dcm:ImagerPixelSpacing']

    for key in spacing_keys:
        raw_spacing = image.info.get(key)
        
        if not raw_spacing:
            continue
            
        try:
            # parse backslash-separated spacing values
            parts = raw_spacing.strip().split('\\')
            
            if len(parts) < 2:
                if logger:
                    logger.warning(f"Invalid spacing format in '{key}': expected at least 2 values, got {len(parts)}")
                continue

            # convert to float (row spacing, column spacing)    
            pixel_spacing_y = float(parts[0])  # Row spacing
            pixel_spacing_x = float(parts[1])  # Column spacing
            
            return key, pixel_spacing_x, pixel_spacing_y, manufacturer, manufacturer_model_name
            
        except (ValueError, TypeError) as e:
            if logger:
                logger.warning(f"Failed to parse '{key}': {e}")
            continue

    # If no valid key found or parsing failed
    if logger:
        logger.debug("No valid pixel spacing metadata found in image")
    
    return None, None, None, manufacturer, manufacturer_model_name

def get_image_info(image_file, response, logger=None):
    
    """
    Extract basic metadata from an image file (DICOM or JPEG).
    DICOM images are read using pydicom.dcmread. Non-DICOM images are opened with PIL.Image.open.
    Falls back to "Unknown" for missing DICOM attributes.
    :param image_file: path to the JPEG image file
    :return: A tuple containing:
        - width (int): Width of the image in pixels.
        - height (int): Height of the image in pixels.
        - color_scheme (str): Color mode of the image (e.g., 'RGB', 'L', etc.).
    """

    info = {
        "file": image_file,
        "file_type": None,
        "width": None,
        "height": None,
        "color_scheme": None,
        "row_spacing": None,
        "col_spacing": None,
        "dpi": None,
    }

    ds = None
    image = None

    if os.path.exists(image_file): 
        _is_dicom = is_dicom(image_file)
        if _is_dicom: 
            ds = pydicom.dcmread(image_file) 
        else: 
            image = Image.open(image_file)
    elif response is not None: 
        bytes_content = BytesIO(response.content)
        try:
            ds = pydicom.dcmread(bytes_content, stop_before_pixels=True)
            _is_dicom = True
        except InvalidDicomError:
            image = Image.open(bytes_content)
            _is_dicom = False
    else: 
        print(f" ERROR: No file or HTTP content found for {image_file}!")
        if logger is not None: 
            logger.error(f"No file or HTTP content found for {image_file}!")
        return info

    if _is_dicom:
        # DICOM info
        info["file_type"] = "DICOM"
        info["width"] = int(ds.Columns)
        info["height"] = int(ds.Rows)
        info["color_scheme"] = getattr(ds, "PhotometricInterpretation", None)
        info["manufacturer"] = getattr(ds, "Manufacturer", None)
        info["manufacturer_model_name"] = getattr(ds, "ManufacturerModelName", None)

        pixel_size = getattr(ds, "PixelSpacing", None)
        if pixel_size is not None:
            pixel_size = [round(x, 3) for x in pixel_size]
            info["row_spacing"], info["col_spacing"] = map(float, pixel_size)
    else:
        # Assume JPEG/PNG/etc.
        info["file_type"] = image.format
        info["width"], info["height"] = image.size
        info["color_scheme"] = image.mode
        info["dpi"] = str(image.info.get("dpi"))
        if image.format == 'PNG':
            spacing_method, info["row_spacing"], info["col_spacing"], info["manufacturer"], info["manufacturer_model_name"] = get_png_pixel_spacing(image)

    return info

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Applies a Gaussian blur to the input image to reduce noise and smoothen the image.

    Args:
        con (sqlite3.Connection, optional): A SQLite database connection for logging the operation. Default is None.
        image (ndarray): The input image as a NumPy array (grayscale or color) on which to apply Gaussian blur.
        image_archive_id (int): A unique identifier for the image in the database for tracking.
        write (bool, optional): If True, logs the operation to the database using the provided connection. Default is True.
        archive (object, optional): A reference to the archive system (could be used for file-based or DB archiving). Default is None.
        kernel_size (tuple, optional): The size of the Gaussian kernel. Default is (5, 5).
        sigma (int, optional): The standard deviation in the X and Y direction for the Gaussian kernel. Default is 0 (auto-calculated).

    Returns:
        ndarray: The resulting image after applying Gaussian blur.

    Raises:
        ValueError: If the input image is None or invalid.
        sqlite3.Error: If a database error occurs during logging.

    Notes:
        Gaussian blur is commonly used for reducing image noise and detail.
        It works by applying a convolution between the image and a Gaussian kernel.
    """
    method_name="gaussian_blur"
    # Validate the input image
    if image is None:
        raise ValueError("Input image is None. Cannot apply Gaussian blur.")

    # Apply Gaussian blur to the image
    result_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return result_image, method_name

def apply_edge_enhancement(image, scale=1.0, delta=0, ddepth=cv2.CV_64F):
    """
    Enhances the edges of the input image using the Laplacian operator, emphasizing regions with sharp transitions.

    Args:
        image (ndarray): The input image as a NumPy array (grayscale or color) to enhance edges.
        scale (float, optional): Scaling factor for the Laplacian gradient values. Default is 1.0.
        delta (int, optional): Value added to the results after applying Laplacian. Default is 0.
        ddepth (int, optional): Desired depth of the destination image. Default is cv2.CV_64F for high precision.

    Returns:
        ndarray: The resulting image with enhanced edges.

    Raises:
        ValueError: If the input image is None or invalid.

    Notes:
        The Laplacian operator is used to detect edges by calculating the second derivative of the image.
        The function subtracts the Laplacian from the original image, enhancing the contrast at edges.
    """
    method_name = "edge_enhancement"
    # Validate the input image
    if image is None:
        raise ValueError("Input image is None. Cannot apply edge enhancement.")

    # Step 1: Apply the Laplacian operator to detect edges
    laplacian = cv2.Laplacian(image, ddepth, scale=scale, delta=delta)

    # Step 2: Subtract the Laplacian from the original image to enhance edges
    enhanced_image = cv2.convertScaleAbs(image - laplacian)

    return enhanced_image, method_name

def crop_image_from_right(image, threshold_factor=20):
    """
    Crops the image from the right based on brightness differences between columns.

    Args:
        image (ndarray): The input image as a NumPy array (grayscale or color).
        threshold_factor (float, optional): The factor that determines when a significant brightness difference triggers the crop.
                                            A higher value makes the cropping more sensitive to brightness differences. Default is 1.0.

    Returns:
        ndarray: The cropped image.
        string: name of the method applied to the image

    Raises:
        ValueError: If the input image is invalid or if no significant brightness difference is found.
    """
    image = image.copy()
    method_name = "crop_scale_right"
    # Validate the input image
    if image is None or len(image.shape) < 2:
        raise ValueError("The input image is invalid.")

    # Compute the average brightness for each column
    average_brightness = image.mean(axis=0)

    # Calculate the difference in brightness between neighboring columns
    diff = np.diff(average_brightness)

    # Find the cutoff point based on the threshold factor
    cutoff_indices = np.where(np.abs(diff) > diff.std() * threshold_factor)

    if cutoff_indices[0].size > 0:
        # Use the first significant brightness difference as the cutoff point
        cutoff_point = cutoff_indices[0][0]
    else:
        # If no significant difference is found, use the full image width
        cutoff_point = image.shape[1]

    # Crop the image from the right up to the cutoff point
    cropped_img = image[:, :cutoff_point]

    return cropped_img, method_name

def crop_image_from_top(image, threshold_factor=1.2, fallback_percent=0.1):
    
    method_name = "crop_scale_top"
    # Validate the input image
    if image is None or len(image.shape) < 2:
        raise ValueError("The input image is invalid.")

    # Analyze the brightness of each row (mean brightness across rows)
    row_brightness = image.mean(axis=1)

    # Calculate the cutoff based on the threshold factor
    cutoff_indices = np.where(row_brightness > row_brightness.mean() * threshold_factor)

    # Check if a threshold row is found
    if cutoff_indices[0].size > 0:
        cutoff_row = cutoff_indices[0][0]  # First row exceeding the brightness threshold
    else:
        # Fallback: Use a fixed percentage of the image height (e.g., the top 10%)
        cutoff_row = int(image.shape[0] * fallback_percent)

    # Crop the image from the top, starting from the found row or the fallback area
    cropped_image = image[cutoff_row:, :]

    return cropped_image,method_name

def detect_image_inversion(image, brightness_threshold=128):
    """
    Determines if an image is likely inverted based on its mean brightness.

    Args:
        image (ndarray): The input image as a NumPy array (grayscale or color).
        brightness_threshold (int, optional): The threshold for determining inversion, where higher mean brightness
                                              suggests inversion. Default is 128.

    Returns:
        bool: True if the image is likely inverted, False otherwise.

    Raises:
        ValueError: If the input image is None or not in grayscale format.

    Notes:
        Inversion is typically detected by evaluating the mean brightness of the image. In a normally exposed image,
        lower mean brightness corresponds to a darker image (background dark, foreground light), while an inverted
        image will have a higher mean brightness.
    """

    # Validate input
    if image is None:
        raise ValueError("Input image is None. Cannot determine inversion.")

    # Ensure the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Calculate the mean brightness directly from the image array
    mean_brightness = np.mean(image)

    # Determine if the image is likely inverted based on brightness
    return mean_brightness > brightness_threshold

def analyze_center_brightness(img_array, window_fraction=0.2):
    """
    Analyze the brightness in the central part of a radiograph.

    :param img_array:
    :param window_fraction: Fraction of the image's width and height to consider as the center.
    :return: Average and median brightness of the central region.
    """

    # Calculate the center window dimensions
    height, width = img_array.shape
    center_x, center_y = width // 2, height // 2
    window_width, window_height = int(width * window_fraction), int(height * window_fraction)

    # Define the central region
    start_x = max(center_x - window_width // 2, 0)
    end_x = min(center_x + window_width // 2, width)
    start_y = max(center_y - window_height // 2, 0)
    end_y = min(center_y + window_height // 2, height)

    # Extract the central region
    central_region = img_array[start_y:end_y, start_x:end_x]

    # Calculate brightness statistics
    average_brightness = np.mean(central_region)
    median_brightness = np.median(central_region)

    return average_brightness, median_brightness

def remove_marker(image, threshold=55):
    
    method_name = "remove_marker"

    # Binarize the image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels < 2:
        raise ValueError("No connected components found in the image.")

    # Sort the components by size (area), ignoring the background
    largest_components = stats[1:, -1].argsort()[::-1] + 1  # Background is component 0

    # Create a mask with only the largest connected component
    mask = np.zeros_like(binary)
    mask[labels == largest_components[0]] = 255

    # Apply the mask to the original image
    cleaned_image = cv2.bitwise_and(image, mask)

    return cleaned_image,method_name

def invert_image(image):
    """
    Inverts the colors of the input image.

    Args:
        image (ndarray): The input image as a NumPy array (grayscale or color).

    Returns:
        ndarray: The inverted image as a NumPy array.
        string: name of the method applied to the image

    Raises:
        ValueError: If the input image is invalid or if it has an unsupported mode.
    """
    method_name ="inversion"
    # Validate the input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("The input image is invalid. Expected a NumPy array.")

    # Convert the image to a PIL Image object
    pil_image = Image.fromarray(image)

    # Ensure the image is in a mode that can be inverted (grayscale or RGB)
    if pil_image.mode not in ["L", "RGB"]:
        raise ValueError(f"Unsupported image mode {pil_image.mode}. Expected 'L' for grayscale or 'RGB' for color.")

    # Invert the image
    inverted_image = ImageOps.invert(pil_image)

    # Convert the inverted image back to a NumPy array
    inverted_image = np.array(inverted_image)

    return inverted_image,method_name

def apply_clahe(image, clahe_clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of the image.

    Args:
        image (ndarray): The input grayscale image as a NumPy array to which CLAHE will be applied.
        clahe_clip_limit (float, optional): Threshold for contrast limiting. Higher values increase contrast. Default is 2.0.
        tile_grid_size (tuple, optional): Size of the grid for histogram equalization. Default is (8, 8), meaning the image is divided into 8x8 tiles.

    Returns:
        ndarray: The resulting image after applying CLAHE for contrast enhancement.

    Raises:
        ValueError: If the input image is None or not a grayscale image.

    Notes:
        CLAHE is particularly useful for enhancing contrast in localized regions while avoiding noise amplification, which is common in global histogram equalization.
    """
    method_name = "clahe"
    # Validate the input image
    if image is None:
        raise ValueError("Input image is None. Cannot apply CLAHE.")

    if len(image.shape) != 2:
        raise ValueError("CLAHE can only be applied to grayscale images. Please provide a single-channel image.")

    # Create CLAHE object with specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the image
    result_image = clahe.apply(image)

    return result_image,method_name

def pad_image_to_square(image, border_width=10):
    """
    @TODO
    Pads the given image to a square shape.

    Parameters:
    - image: The image to be padded.
    - border_width: Number of pixels to use from the borders to calculate the padding color.

    Returns:
    - padded_image: The padded image.
    """

    # if len(image.shape) == 2:
    #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Ursprüngliche Abmessungen des Bildes bestimmen
    original_height, original_width = image.shape[:2]

    # Bestimme, welche Seite länger ist
    if original_height > original_width:
        padding_size = (original_height - original_width) // 2
        left_padding = right_padding = padding_size
        top_padding = bottom_padding = 0
        if (original_height - original_width) % 2 != 0:  # Ungerade Differenz
            right_padding += 1
    else:
        padding_size = (original_width - original_height) // 2
        top_padding = bottom_padding = padding_size
        left_padding = right_padding = 0
        if (original_width - original_height) % 2 != 0:  # Ungerade Differenz
            bottom_padding += 1

    # Bestimme die Randfarbe unter Berücksichtigung der border_width
    if top_padding > 0:  # Vertikales Padding
        edge_pixels = np.concatenate([image[:border_width, :], image[-border_width:, :]])  # Die ersten und letzten `border_width` Zeilen
    else:  # Horizontales Padding
        edge_pixels = np.concatenate([image[:, :border_width], image[:, -border_width:]])  # Die ersten und letzten `border_width` Spalten

    # Berechne den Durchschnittswert der Randfarbe
    avg_color = np.mean(edge_pixels, axis=(0, 1)).astype(np.uint8)  # Mittelt über Höhe und Breite

    # Hintergrundbilder für verschiedene Padding-Bereiche erstellen
    if left_padding > 0 or right_padding > 0:
        vertical_pad = np.full((original_height, left_padding), avg_color, dtype=np.uint8)
        image = np.hstack([vertical_pad, image, np.full((original_height, right_padding), avg_color, dtype=np.uint8)])
    if top_padding > 0 or bottom_padding > 0:
        horizontal_pad = np.full((top_padding, image.shape[1]), avg_color, dtype=np.uint8)
        image = np.vstack([horizontal_pad, image, np.full((bottom_padding, image.shape[1]), avg_color, dtype=np.uint8)])

    return image

def resize_image_to_array(image_file_path, target_size, img=None):
    """
    Load an image from the given path, resize it, convert it to grayscale, and return it as a NumPy array.

    The image is resized to target_size, converted to grayscale, and transformed into a NumPy array.
    This method uses OpenCV for handling image operations and outputs a 4D array.
    Raises FileNotFoundError if the specified image path does not exist.
    :param img:
    :param image_file_path:
    :param target_size:
    :return: A NumPy array representing the processed image. The array is 4D with an additional batch dimension at index 0.
    """

    if img is None:

        # Check if the provided image path exists
        if not os.path.exists(image_file_path):
            raise FileNotFoundError(f"The specified image file does not exist: {image_file_path}")

        # Load the image in grayscale (0 = grayscale mode)
        img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load the image from path: {image_file_path}")

    # Resize the image to 150x150 pixels
    img = cv2.resize(img, target_size)

    # Expand dimensions to match the expected 4D shape (batch size, height, width, channels)
    img = np.expand_dims(img, axis=(0, -1))  # Shape becomes (1, 150, 150, 1)

    return img

def apply_brightness(image, target_brightness=20):
    """
    @TODO
    Adjusts the brightness of an image to match a target brightness level.

    Args:
        image (ndarray): The input image as a NumPy array (grayscale or color).
        target_brightness (int, optional): The desired average brightness of the image. Default is 128.

    Returns:
        ndarray: The brightness-adjusted image as a NumPy array.
        string: name of the method applied to the image

    Raises:
        ValueError: If the input image is invalid or if the target brightness is outside the valid range (0-255).
    """

    method_name = "adjust_brightness"
    # Validate the input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("The input image is invalid. Expected a NumPy array.")

    if target_brightness < 0 or target_brightness > 255:
        raise ValueError("Target brightness must be between 0 and 255.")

    # Calculate the average brightness of the image
    avg_brightness = np.mean(image)

    # Calculate the delta to adjust the brightness
    delta = target_brightness - avg_brightness

    # Adjust the image brightness
    adjusted_image = image + delta

    # Clip the values to ensure they remain between 0 and 255
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image,method_name

def rotate_image(image, angle, border_width=None, background_color=None):
    """
    Rotates an image by a specified angle while preserving the entire image content.
    
    :param image: The input image to be rotated (NumPy array).
    :param angle: The rotation angle in degrees. Positive values rotate the image counter-clockwise.
    :param border_width: Number of pixels to use from the borders to calculate the background color. If None, defaults to 10.
    :param background_color: The color to fill the areas outside the original image. If None, it is calculated from the image borders.
    
    :return: The rotated image with preserved content and filled background (NumPy array).
    """
    if background_color is None:
        background_color = measure_image_border_color(image, border_width=border_width)

    h, w = image.shape[:2]
    
    # rotated
    max_side = max(h, w)

    center = (max_side // 2, max_side // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, M, (max_side, max_side), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=background_color) 
    
    return result

def measure_image_border_color(image, border_width=10):
    """
    Measures the average color of the borders of an image to determine a suitable background color for padding or filling.
    
    :param image: The input image as a NumPy array (grayscale or color).
    :param border_width: The width of the border area to consider for color measurement (default: 10 pixels).
    
    :return: The average color of the borders as a NumPy array (for color images) or an integer (for grayscale images).
    """
    h, w = image.shape[:2]

    # extract border regions
    top_border = image[:border_width, :]  # oberer Rand
    bottom_border = image[-border_width:, :]  # unterer Rand
    left_border = image[:, :border_width]  # linker Rand
    right_border = image[:, -border_width:]  # rechter Rand

    # first combine the top and bottom borders along the height (axis 0)
    top_bottom_border = np.concatenate([top_border, bottom_border], axis=0)

    # then combine the left and right borders along the width (axis 1)
    left_right_border = np.concatenate([left_border, right_border], axis=1)

    # combine all border pixels for the final average color calculation
    if image.ndim == 3:  # color image
        border_color = np.mean(np.concatenate([top_bottom_border, left_right_border], axis=None), axis=0).astype(np.uint8)
        return border_color
    elif image.ndim == 2:  # grayscale image
        # mean value of the top and bottom borders
        mean_top_bottom = np.mean(top_bottom_border)
        # mean value of the left and right borders
        mean_left_right = np.mean(left_right_border)
        # average of both means to account for border_width influence correctly
        border_color = (mean_top_bottom + mean_left_right) / 2
        return int(border_color)
    else:
        raise ValueError("Unexpected image format")

def analyze_maus_brightness_median(image, bright_pixel_threshold=200, dark_pixel_threshold=20, median_threshold=180, bright_ratio_threshold=50):
    """
    Analyzes the brightness of a grayscale mouse image based on the median and ratio of bright pixels.

    :param image: Input grayscale image of the mouse as a NumPy array.
    :param bright_pixel_threshold: Threshold value for identifying bright pixels (default: 200).
    :param dark_pixel_threshold: Threshold value for identifying dark pixels (default: 20).
    :param median_threshold: Threshold value for the median brightness (default: 180).
    :param bright_ratio_threshold: Threshold value for the ratio of bright pixels (default: 50).

    :return: True if the image is considered too bright (overexposed), False otherwise.
    """
    
    too_bright = False

    # remove background (exclude very dark pixels)
    non_zero_pixels = image[image > dark_pixel_threshold]

    if non_zero_pixels.size == 0:
        # print("No non-black pixels found in the image.")
        return False

    # compute statistics
    median_val = np.median(non_zero_pixels)

    # identify very bright pixels and calculate their ratio
    bright_pixels = non_zero_pixels[non_zero_pixels > bright_pixel_threshold]
    bright_ratio = len(bright_pixels) / len(non_zero_pixels) * 100

    # heuristic overexposure criterion
    if median_val > median_threshold and bright_ratio > bright_ratio_threshold:
        # print("Overexposed!")
        too_bright = True

    return too_bright

def detect_global_overexposure(image, bg_thresh=150, obj_thresh=160):
    """
    The function defines background and object regions based on a border around the image. 
    It calculates the mean brightness of both regions and determines if the image is globally overexposed based on the specified thresholds.

    :param image: Input image as a NumPy array (grayscale or color).
    :param bg_thresh: Threshold value for background brightness.
    :param obj_thresh: Threshold value for object brightness.

    :return: True if the image is globally overexposed, False otherwise.    
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    border = int(0.15 * min(h, w))

    # Masken definieren
    bg_mask = np.zeros_like(image, dtype=bool)
    bg_mask[:border, :] = True
    bg_mask[-border:, :] = True
    bg_mask[:, :border] = True
    bg_mask[:, -border:] = True

    obj_mask = ~bg_mask

    bg_pixels = image[bg_mask]
    obj_pixels = image[obj_mask]

    mean_bg = np.mean(bg_pixels)
    mean_obj = np.mean(obj_pixels)

    if (
        mean_bg > bg_thresh and
        mean_obj > obj_thresh
    ):
        # print("Global overexposed!")
        return True

    return False

def binarize_images(x, threshold=0.4, logger=None):
    """
    Normalizes and binarizes an image or a batch of images.
    
    :param x: Input image or batch of images as a NumPy array with pixel values in the range [0, 255].
    :param threshold: Threshold value for binarization (default: 0.4).
    :param logger: Logger object for logging messages (default: None).

    :return: Binarized image or batch of images as a NumPy array with pixel values of 0 or 1.
    """

    # Ensure the input is a NumPy array
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Ensure the values are in the expected range [0, 255]
    if np.any(x < 0) or np.any(x > 255):
        raise ValueError("Input array must contain pixel values in the range [0, 255].")

    # Normalize pixel values to the range [0, 1]
    x = x / 255.0

    # Binarize the image: pixels >= threshold become 1, and pixels < threshold become 0
    x = np.where(x >= threshold, 1, 0)

    return x
