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
Utilities for image metadata extraction and preprocessing.

The module contains helpers for:
- detecting downloaded image formats and extracting metadata from DICOM and
  common raster files,
- applying contrast, blur, inversion, cropping, padding, and rotation
  transforms, and
- computing simple brightness and exposure heuristics used in downstream image
  quality checks.

Most functions operate on NumPy arrays, while the metadata helpers work with
PIL images, local files, or HTTP response content.
"""

def detect_file_format(response):
    """
    Detect an image file format from an HTTP response.

    Parameters
    ----------
    response : requests.Response
        Response whose headers and content should be inspected.

    Returns
    -------
    str
        Detected format name such as `"JPEG"`, `"PNG"`, `"BMP"`, `"DICOM"`,
        or `"Unknown"` if no parser succeeds.

    Notes
    -----
    Detection first consults the `Content-Type` header, then falls back to PIL
    and finally to `pydicom`.
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
    Extract embedded pixel-spacing metadata from a PNG image.

    Parameters
    ----------
    image : PIL.Image.Image
        PNG image whose `info` and `text` dictionaries may contain DICOM-style
        metadata fields.
    logger : logging.Logger, optional
        Logger used for parse warnings and debug output.

    Returns
    -------
    tuple[str | None, float | None, float | None, str | None, str | None]
        The spacing key that matched, column spacing, row spacing,
        manufacturer, and manufacturer model name. Spacing values are `None`
        when the expected metadata keys are missing or malformed.

    Raises
    ------
    TypeError
        Raised when `image` is not a PIL image.
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
    Extract image metadata from a local file or downloaded response body.

    Parameters
    ----------
    image_file : str | os.PathLike
        Path to a local image file. If the file exists, it is preferred over
        the HTTP response content.
    response : requests.Response | None
        Optional HTTP response used when `image_file` does not yet exist on
        disk.
    logger : logging.Logger, optional
        Logger used for error reporting when neither source is available.

    Returns
    -------
    dict
        Metadata dictionary containing fields such as `file`, `file_type`,
        `width`, `height`, `color_scheme`, spacing information, and optional
        manufacturer metadata. PNG files may additionally expose spacing values
        embedded in textual metadata.

    Notes
    -----
    DICOM images are read with `pydicom`, while non-DICOM files are opened with
    PIL. When no local file exists and `response` is `None`, the function logs
    the error and returns a dictionary with `None`-valued metadata fields.
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
    Apply Gaussian blur to reduce image noise.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    kernel_size : tuple[int, int], optional
        Size of the Gaussian kernel.
    sigma : float, optional
        Standard deviation used by OpenCV. A value of `0` lets OpenCV infer it
        from `kernel_size`.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Blurred image and the method label `"gaussian_blur"`.

    Raises
    ------
    ValueError
        Raised when `image` is `None`.
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
    Enhance edges with a Laplacian-based sharpening step.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    scale : float, optional
        Scaling factor passed to `cv2.Laplacian`.
    delta : float, optional
        Value added to the Laplacian result.
    ddepth : int, optional
        OpenCV output depth used for the Laplacian computation.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Edge-enhanced image and the method label `"edge_enhancement"`.

    Raises
    ------
    ValueError
        Raised when `image` is `None`.
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
    Crop an image from the right using column-brightness changes.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    threshold_factor : float, optional
        Multiplier applied to the standard deviation of adjacent-column
        brightness differences to determine the crop trigger.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Cropped image and the method label `"crop_scale_right"`.

    Raises
    ------
    ValueError
        Raised when the input is missing or has fewer than two dimensions.
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
    Crop an image from the top using row-brightness analysis.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    threshold_factor : float, optional
        Multiplier applied to the mean row brightness to find the first row
        that marks the crop start.
    fallback_percent : float, optional
        Fraction of the image height to skip when no threshold crossing is
        detected.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Cropped image and the method label `"crop_scale_top"`.

    Raises
    ------
    ValueError
        Raised when the input is missing or has fewer than two dimensions.
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
    Heuristically detect whether a grayscale image looks inverted.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    brightness_threshold : float, optional
        Mean-brightness threshold above which the image is treated as likely
        inverted.

    Returns
    -------
    bool
        `True` when the mean brightness exceeds `brightness_threshold`.

    Raises
    ------
    ValueError
        Raised when the input is missing or is not a 2D grayscale image.
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
    Measure brightness statistics in the image center.

    Parameters
    ----------
    img_array : numpy.ndarray
        Input 2D grayscale image.
    window_fraction : float, optional
        Fraction of the image width and height used to define the centered
        analysis window.

    Returns
    -------
    tuple[float, float]
        Mean and median brightness of the centered region.
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
    Remove bright markers by keeping only the largest connected component.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    threshold : int, optional
        Threshold used to binarize the image before component analysis.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Cleaned image and the method label `"remove_marker"`.

    Raises
    ------
    ValueError
        Raised when no foreground connected component is found.
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
    Invert a grayscale or RGB image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or RGB image.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Inverted image and the method label `"inversion"`.

    Raises
    ------
    ValueError
        Raised when the input is missing, is not a NumPy array, or uses an
        unsupported PIL mode after conversion.
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
    Apply CLAHE contrast enhancement to a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    clahe_clip_limit : float, optional
        Clip limit passed to OpenCV's CLAHE implementation.
    tile_grid_size : tuple[int, int], optional
        Tile size used for local histogram equalization.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Contrast-enhanced image and the method label `"clahe"`.

    Raises
    ------
    ValueError
        Raised when the image is missing or is not single-channel.
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
    Pad an image symmetrically until width and height match.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    border_width : int, optional
        Width of the edge strip used to estimate the padding color.

    Returns
    -------
    numpy.ndarray
        Square image padded with an edge-color estimate. If the original aspect
        ratio differs by an odd number of pixels, the extra pixel is added to
        the right or bottom edge.
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
    """
    Translate the foreground object so it is centered on a square canvas.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale or color image containing a single foreground object.
    use_replicate : bool, optional
        If `True`, extend the border with `cv2.BORDER_REPLICATE`. Otherwise use
        a constant background estimated from border pixels.
    center_mode : {"bbox", "centroid"}, optional
        Strategy used to determine the object's center point.
    output_mode : {"fixed_max", "fit_centered", "fixed_then_crop"}, optional
        Controls the output canvas size:
        - `fixed_max`: return a square with side length `max(height, width)`.
        - `fit_centered`: grow the canvas until the translated object fits.
        - `fixed_then_crop`: fit first, then crop back to `max(height, width)`.

    Returns
    -------
    numpy.ndarray
        Centered square image.

    Raises
    ------
    ValueError
        Raised when Otsu thresholding does not detect any foreground pixels.
    """
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
    Load an image from disk, resize it, and return it as a batch array.

    Parameters
    ----------
    image_file_path : str | os.PathLike
        Path to the input image.
    target_size : tuple[int, int]
        Target `(width, height)` passed to PIL resize.

    Returns
    -------
    numpy.ndarray
        Array of shape `(1, height, width, 1)` when the file exists, or an
        empty array when the path is missing. The image is converted to
        grayscale before conversion with `img_to_array`.
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
    Shift image brightness toward a target mean value.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    target_brightness : float, optional
        Desired mean brightness in the range `[0, 255]`.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Brightness-adjusted image and the method label `"adjust_brightness"`.

    Raises
    ------
    ValueError
        Raised when the input is invalid or `target_brightness` is outside
        `[0, 255]`.
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
    Rotate an image on a square output canvas.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to rotate.
    angle : float
        Counter-clockwise rotation angle in degrees.
    border_width : int | None, optional
        Border width used when estimating the background color. Ignored when
        `background_color` is provided.
    background_color : int | sequence[int] | None, optional
        Fill value used outside the original image. When omitted, the value is
        estimated from the image borders.

    Returns
    -------
    numpy.ndarray
        Rotated image on a square canvas with side length `max(height, width)`.
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
    Estimate a representative border color from an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image.
    border_width : int, optional
        Width of the border strip sampled from all four edges.

    Returns
    -------
    numpy.uint8 | int
        Average border value. Grayscale inputs return an integer; color inputs
        return a NumPy scalar/array compatible with OpenCV border fills.

    Raises
    ------
    ValueError
        Raised when the input dimensionality is not recognized.
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
    Flag likely overexposure from grayscale brightness heuristics.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    bright_pixel_threshold : float, optional
        Threshold above which a pixel counts as bright.
    dark_pixel_threshold : float, optional
        Threshold below which pixels are treated as background and ignored.
    median_threshold : float, optional
        Median brightness threshold for overexposure detection.
    bright_ratio_threshold : float, optional
        Minimum percentage of bright pixels required to call the image too
        bright.

    Returns
    -------
    bool
        `True` when both the median and bright-pixel-ratio heuristics indicate
        overexposure.
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
    Detect global overexposure from border and interior brightness.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image. Color inputs are converted to grayscale
        first.
    bg_thresh : float, optional
        Mean brightness threshold for the border region.
    obj_thresh : float, optional
        Mean brightness threshold for the interior object region.

    Returns
    -------
    bool
        `True` when both border and interior brightness exceed their
        thresholds.
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
    Normalize image data to `[0, 1]` and threshold it to a binary mask.

    Parameters
    ----------
    x : numpy.ndarray
        Input image or image batch with values in the range `[0, 255]`.
    threshold : float, optional
        Threshold applied after normalization to `[0, 1]`.
    logger : logging.Logger, optional
        Currently unused placeholder for callers that pass a logger interface.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as `x` containing only `0` and `1`.

    Raises
    ------
    ValueError
        Raised when `x` is not a NumPy array or contains values outside the
        expected pixel range.
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
