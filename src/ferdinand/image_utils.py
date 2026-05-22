import pydicom
import os
import cv2

import numpy as np
import PIL.ImageOps as ImageOps

from pydicom.errors import InvalidDicomError
from pydicom.misc import is_dicom
from tensorflow.keras.utils import img_to_array
from PIL import Image
from io import BytesIO

"""
This module provides utility functions for processing and analyzing images, including DICOM and common image formats.
"""

def detect_file_format(response):
    """
    Detect image file format from an HTTP response by checking the Content-Type header and attempting to read the content with PIL and pydicom.
    
    :param response: requests.Response object from requests.get()
    :return: Detected file format, e.g. "JPEG", "PNG", "DICOM", or "Unknown".
    """
    # 1. check HTTP Content-Type header
    content_type = response.headers.get("Content-Type", "").lower()
    if "jpeg" in content_type:
        return "JPEG"
    elif "png" in content_type:
        return "PNG"
    elif "dicom" in content_type:
        return "DICOM"
    elif "bmp" in content_type:
        return "BMP"

    # 2. try with PIL (JPEG, PNG, etc.)
    try:
        img = Image.open(BytesIO(response.content))
        return img.format  # e.g. "JPEG", "PNG"
    except Exception:
        pass

    # 3. try with pydicom
    try:
        _ = pydicom.dcmread(BytesIO(response.content))
        return "DICOM"
    except Exception:
        pass

    return "Unknown"

def get_png_pixel_spacing(image, logger=None):
    """
    Extract metadata (size, spacing, DPI, manufacturer) from DICOM/PNG/JPEG.

    :param image: A PIL Image object containing the image and its metadata.
    :param logger: Optional logger for logging warnings and errors.

    :return: A tuple containing:
        - spacing_key (str): The key used to extract pixel spacing (e.g., 'dcm:PixelSpacing' or 'dcm:ImagerPixelSpacing').
        - pixel_spacing_x (float): The pixel spacing in the x-direction (column spacing).
        - pixel_spacing_y (float): The pixel spacing in the y-direction (row spacing).
        - manufacturer (str or None): The manufacturer of the imaging device, if available in the metadata.
        - manufacturer_model_name (str or None): The model name of the imaging device, if available in the metadata.
    """
    # validate input type
    if not isinstance(image, Image.Image):
        error_msg = f"Expected PIL Image object, got {type(image).__name__}"
        if logger:
            logger.error(error_msg)
        raise TypeError(error_msg)

    metadata = {}
    metadata.update(getattr(image, "info", {}))
    metadata.update(getattr(image, "text", {}))

    # print(f"Available metadata keys: {list(metadata.keys())}")
    
    # extract manufacturer metadata if available
    manufacturer = metadata.get('dcm:Manufacturer')
    manufacturer_model_name = metadata.get("dcm:Manufacturer'sModelName")
    
    spacing_keys = ['dcm:PixelSpacing', 'dcm:ImagerPixelSpacing']

    for key in spacing_keys:
        raw_spacing = metadata.get(key)
        
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

    # if no valid key found or parsing failed
    if logger:
        logger.debug("No valid pixel spacing metadata found in image")
    
    return None, None, None, manufacturer, manufacturer_model_name

def get_image_info(image_file, response, logger=None):
    """
    Extract basic metadata from an image file (DICOM or JPEG/PNG).
    DICOM images are read using pydicom.dcmread. Non-DICOM images are opened with PIL.Image.open.
    Falls back to "Unknown" for missing DICOM attributes.
    
    :param image_file: path to the JPEG image file
    :param response: HTTP response containing image data
    :param logger: Optional logger for logging warnings and errors
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
        # assume JPEG/PNG/etc.
        info["file_type"] = image.format
        info["width"], info["height"] = image.size
        info["color_scheme"] = image.mode
        info["dpi"] = str(image.info.get("dpi"))
        if image.format == 'PNG':
            spacing_method, info["row_spacing"], info["col_spacing"], info["manufacturer"], info["manufacturer_model_name"] = get_png_pixel_spacing(image)
            # print(spacing_method, info["row_spacing"], info["col_spacing"], info["manufacturer"], info["manufacturer_model_name"])

    return info

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Applies a Gaussian blur to the input image to reduce noise and smoothen the image.

    :param image: The input image as a NumPy array (grayscale or color) on which to apply Gaussian blur.
    :param kernel_size: The size of the Gaussian kernel (default: (5, 5)).
    :param sigma: The standard deviation in the X and Y direction for the Gaussian kernel (default: 0, which means it is calculated from the kernel size).

    :return: The resulting image after applying Gaussian blur as a NumPy array and the method name.
    """
    method_name="gaussian_blur"
    # validate the input image
    if image is None:
        raise ValueError("Input image is None. Cannot apply Gaussian blur.")

    # apply Gaussian blur to the image
    result_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return result_image, method_name

def apply_edge_enhancement(image, scale=1.0, delta=0, ddepth=cv2.CV_64F):
    """
    Enhances the edges of the input image using the Laplacian operator, emphasizing regions with sharp transitions.

    :param image: The input image as a NumPy array (grayscale or color) to enhance edges.
    :param scale: Scaling factor for the Laplacian gradient values (default: 1.0).
    :param delta: Value added to the results after applying Laplacian (default: 0).
    :param ddepth: Desired depth of the destination image (default: cv2.CV_64F for high precision).

    :return: The resulting image with enhanced edges as a NumPy array and the method name.
    """
    method_name = "edge_enhancement"
    # validate the input image
    if image is None:
        raise ValueError("Input image is None. Cannot apply edge enhancement.")

    # step 1: apply the Laplacian operator to detect edges
    laplacian = cv2.Laplacian(image, ddepth, scale=scale, delta=delta)

    # step 2: subtract the Laplacian from the original image to enhance edges
    enhanced_image = cv2.convertScaleAbs(image - laplacian)

    return enhanced_image, method_name

def crop_image_from_right(image, threshold_factor=20):
    """
    Crops the image from the right based on brightness differences between columns. 
    The function analyzes the average brightness of each column and identifies significant drops in brightness 
    to determine where to crop the image from the right side. 
    The threshold_factor parameter controls the sensitivity of the cropping, with higher values making it 
    more likely to crop based on smaller brightness differences.

    :param image: The input image as a NumPy array (grayscale or color) to be cropped.
    :param threshold_factor: The factor that determines when a significant brightness difference triggers the crop.
    :return: The cropped image as a NumPy array and the method name.
    """
    image = image.copy()
    method_name = "crop_scale_right"
    # validate the input image
    if image is None or len(image.shape) < 2:
        raise ValueError("The input image is invalid.")

    # compute the average brightness for each column
    average_brightness = image.mean(axis=0)

    # calculate the difference in brightness between neighboring columns
    diff = np.diff(average_brightness)

    # find the cutoff point based on the threshold factor
    cutoff_indices = np.where(np.abs(diff) > diff.std() * threshold_factor)

    if cutoff_indices[0].size > 0:
        # use the first significant brightness difference as the cutoff point
        cutoff_point = cutoff_indices[0][0]
    else:
        # if no significant difference is found, use the full image width
        cutoff_point = image.shape[1]

    # crop the image from the right up to the cutoff point
    cropped_img = image[:, :cutoff_point]

    return cropped_img, method_name

def crop_image_from_top(image, threshold_factor=1.2, fallback_percent=0.1):
    """
    Crops the image from the top based on brightness differences between rows.

    :param image: The input image as a NumPy array (grayscale or color) to be cropped.
    :param threshold_factor: The factor that determines when a significant brightness difference triggers the crop. Higher values make it more likely to crop based on smaller brightness differences.
    :param fallback_percent: The percentage of the image height to use as a fallback area if no significant brightness difference is found.
    :return: The cropped image as a NumPy array and the method name.
    """
    method_name = "crop_scale_top"
    # validate the input image
    if image is None or len(image.shape) < 2:
        raise ValueError("The input image is invalid.")

    # analyze the brightness of each row (mean brightness across rows)
    row_brightness = image.mean(axis=1)

    # calculate the cutoff based on the threshold factor
    cutoff_indices = np.where(row_brightness > row_brightness.mean() * threshold_factor)

    # check if a threshold row is found
    if cutoff_indices[0].size > 0:
        cutoff_row = cutoff_indices[0][0]  # First row exceeding the brightness threshold
    else:
        # fallback: use a fixed percentage of the image height (e.g., the top 10%)
        cutoff_row = int(image.shape[0] * fallback_percent)

    # crop the image from the top, starting from the found row or the fallback area
    cropped_image = image[cutoff_row:, :]

    return cropped_image,method_name

def detect_image_inversion(image, brightness_threshold=128):
    """
    Detects whether an image is likely inverted based on its mean brightness. 
    In a normally exposed image, lower mean brightness corresponds to a darker image (background dark, foreground light), 
    while an inverted image will have a higher mean brightness. 
    The function uses a specified brightness threshold to determine if the image is likely inverted.

    :param image: The input image as a NumPy array (grayscale or color) to be analyzed for inversion.
    :param brightness_threshold: The threshold for determining inversion, where higher mean brightness suggests inversion. Default is 128.
    :return: A boolean value indicating whether the image is likely inverted (True) or not (False).
    """
    # validate input
    if image is None:
        raise ValueError("Input image is None. Cannot determine inversion.")

    # ensure the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # calculate the mean brightness directly from the image array
    mean_brightness = np.mean(image)

    # determine if the image is likely inverted based on brightness
    return mean_brightness > brightness_threshold

def analyze_center_brightness(img_array, window_fraction=0.2):
    """
    Analyzes the brightness of the central region of an image by calculating the average and median brightness within a defined window around the center.

    :param img_array: The input image as a NumPy array (grayscale or color).
    :param window_fraction: Fraction of the image's width and height to consider as the center.
    :return: Average and median brightness of the central region.
    """
    # calculate the center window dimensions
    height, width = img_array.shape
    center_x, center_y = width // 2, height // 2
    window_width, window_height = int(width * window_fraction), int(height * window_fraction)

    # define the central region
    start_x = max(center_x - window_width // 2, 0)
    end_x = min(center_x + window_width // 2, width)
    start_y = max(center_y - window_height // 2, 0)
    end_y = min(center_y + window_height // 2, height)

    # extract the central region
    central_region = img_array[start_y:end_y, start_x:end_x]

    # calculate brightness statistics
    average_brightness = np.mean(central_region)
    median_brightness = np.median(central_region)

    return average_brightness, median_brightness

def remove_marker(image, threshold=55):
    """
    Removes markers from the image by identifying and isolating the largest connected component based on a brightness threshold.

    :param image: The input image as a NumPy array (grayscale) from which to remove markers.
    :param threshold: The brightness threshold used to binarize the image and identify markers (default: 55).
    :return: The cleaned image with markers removed as a NumPy array and the method name.
    """
    method_name = "remove_marker"

    # binarize the image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels < 2:
        raise ValueError("No connected components found in the image.")

    # sort the components by size (area), ignoring the background
    largest_components = stats[1:, -1].argsort()[::-1] + 1  # Background is component 0

    # create a mask with only the largest connected component
    mask = np.zeros_like(binary)
    mask[labels == largest_components[0]] = 255

    # apply the mask to the original image
    cleaned_image = cv2.bitwise_and(image, mask)

    return cleaned_image,method_name

def invert_image(image):
    """
    Inverts the pixel values of the input image, effectively creating a negative of the image.

    :param image: The input image as a NumPy array (grayscale or color) to be inverted.
    :return: The inverted image as a NumPy array and the method name.
    """
    method_name ="inversion"
    # validate the input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("The input image is invalid. Expected a NumPy array.")

    # convert the image to a PIL Image object
    pil_image = Image.fromarray(image)

    # ensure the image is in a mode that can be inverted (grayscale or RGB)
    if pil_image.mode not in ["L", "RGB"]:
        raise ValueError(f"Unsupported image mode {pil_image.mode}. Expected 'L' for grayscale or 'RGB' for color.")

    # invert the image
    inverted_image = ImageOps.invert(pil_image)

    # convert the inverted image back to a NumPy array
    inverted_image = np.array(inverted_image)

    return inverted_image,method_name

def apply_clahe(image, clahe_clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of the input image.

    :param image: The input image as a NumPy array (grayscale) to which CLAHE will be applied.
    :param clahe_clip_limit: The clip limit for CLAHE (default: 2.0). Higher values give more contrast.
    :param tile_grid_size: The size of the grid for histogram equalization (default: (8, 8)).
    :return: The resulting image after applying CLAHE for contrast enhancement and the method name.
    """
    method_name = "clahe"
    # validate the input image
    if image is None:
        raise ValueError("Input image is None. Cannot apply CLAHE.")

    if len(image.shape) != 2:
        raise ValueError("CLAHE can only be applied to grayscale images. Please provide a single-channel image.")

    # create CLAHE object with specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)

    # apply CLAHE to the image
    result_image = clahe.apply(image)

    return result_image,method_name

def pad_image_to_square(image, border_width=10):
    """
    Pads the input image to make it square by adding borders of a specified width. 
    The padding color is determined based on the average color of the image borders.
    :param image: The input image as a NumPy array (grayscale or color) to be padded.
    :param border_width: The width of the border area to consider for calculating the average color (default: 10 pixels).
    :return: The padded image as a NumPy array.
    """

    # if len(image.shape) == 2:
    #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # original height and width of the image
    original_height, original_width = image.shape[:2]

    # compute the padding needed to make the image square
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

    # calculate the average color of the borders based on the specified border width
    if top_padding > 0:  # Vertikales Padding
        edge_pixels = np.concatenate([image[:border_width, :], image[-border_width:, :]])  
    else:  # horizontal padding
        edge_pixels = np.concatenate([image[:, :border_width], image[:, -border_width:]])  

    # compute the average color of the edge pixels
    avg_color = np.mean(edge_pixels, axis=(0, 1)).astype(np.uint8)  

    # generate the padding and concatenate it to the original image
    if left_padding > 0 or right_padding > 0:
        vertical_pad = np.full((original_height, left_padding), avg_color, dtype=np.uint8)
        image = np.hstack([vertical_pad, image, np.full((original_height, right_padding), avg_color, dtype=np.uint8)])
    if top_padding > 0 or bottom_padding > 0:
        horizontal_pad = np.full((top_padding, image.shape[1]), avg_color, dtype=np.uint8)
        image = np.vstack([horizontal_pad, image, np.full((bottom_padding, image.shape[1]), avg_color, dtype=np.uint8)])

    return image

def center_mouse_on_square(
    img,
    use_replicate=False,
    center_mode="bbox",       # "bbox" | "centroid"
    output_mode="fixed_then_crop",  # "fixed_max" | "fit_centered" | "fixed_then_crop"
):
    h, w = img.shape[:2]
    fixed_size = max(h, w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No object found.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if center_mode == "centroid":
        mouse_cx = int(xs.mean())
        mouse_cy = int(ys.mean())
    else:
        mouse_cx = (x_min + x_max) // 2
        mouse_cy = (y_min + y_max) // 2

    if output_mode == "fixed_max":
        work_size = fixed_size
    else:
        half_size = max(
            mouse_cx,
            w - 1 - mouse_cx,
            mouse_cy,
            h - 1 - mouse_cy
        )
        work_size = int(2 * half_size + 1)

    target_cx = work_size // 2
    target_cy = work_size // 2
    tx = target_cx - mouse_cx
    ty = target_cy - mouse_cy

    M = np.float32([[1, 0, tx], [0, 1, ty]])

    if use_replicate:
        centered = cv2.warpAffine(
            img,
            M,
            (work_size, work_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        border_pixels = np.concatenate([
            gray[:10, :].ravel(),
            gray[-10:, :].ravel(),
            gray[:, :10].ravel(),
            gray[:, -10:].ravel()
        ])
        bg = int(np.median(border_pixels))
        value = bg if img.ndim == 2 else [bg] * img.shape[2]

        centered = cv2.warpAffine(
            img,
            M,
            (work_size, work_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=value
        )

    if output_mode != "fixed_then_crop":
        return centered

    start = (work_size - fixed_size) // 2
    end = start + fixed_size
    cropped = centered[start:end, start:end]

    return cropped

def get_image_as_array(image_file_path, target_size):
    """
    Resizes the image to the specified target size, converts it to grayscale, and returns it as a NumPy array.

    :param image_file_path: The file path to the input image.
    :param target_size: A tuple specifying the desired output size (width, height) for the image.
    :return: A NumPy array containing the processed image data.
    """
    ret_val = []
    if os.path.exists(image_file_path):
        with Image.open(image_file_path) as img:
            img = img.resize(target_size)
            img = img.convert('L')  # convert the image to grayscale
            x = img_to_array(img)
            ret_val.append(x)
    ret_val = np.array(ret_val)
    return ret_val

def apply_brightness(image, target_brightness=20):
    """
    Adjusts the brightness of an image to match a target brightness level.

    :param image: The input image as a NumPy array (grayscale or color).
    :param target_brightness: The desired average brightness of the image. Default is 20.
    :param logger: Optional logger for logging warnings and errors.
    :return: The brightness-adjusted image as a NumPy array and the method name.
    """
    method_name = "adjust_brightness"
    # validate the input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("The input image is invalid. Expected a NumPy array.")

    if target_brightness < 0 or target_brightness > 255:
        raise ValueError("Target brightness must be between 0 and 255.")

    # calculate the average brightness of the image
    avg_brightness = np.mean(image)

    # calculate the delta to adjust the brightness
    delta = target_brightness - avg_brightness

    # adjust the image brightness
    adjusted_image = image + delta

    # clip the values to ensure they remain between 0 and 255
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

def analyze_mouse_brightness_median(image, bright_pixel_threshold=200, dark_pixel_threshold=20, median_threshold=180, bright_ratio_threshold=50):
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

    # define masks for background and object regions
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

    # ensure the input is a NumPy array
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # ensure the values are in the expected range [0, 255]
    if np.any(x < 0) or np.any(x > 255):
        raise ValueError("Input array must contain pixel values in the range [0, 255].")

    # normalize pixel values to the range [0, 1]
    x = x / 255.0

    # binarize the image: pixels >= threshold become 1, and pixels < threshold become 0
    x = np.where(x >= threshold, 1, 0)

    return x