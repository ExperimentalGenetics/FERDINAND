
import pydicom
import os

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

def get_png_pixel_spacing(image_file):

    """
    Extract pixel_spacing_x and pixel_spacing_y from a metadata dictionary.
    Tries 'dcm:PixelSpacing' first, then 'dcm:ImagerPixelSpacing'.
    Returns (pixel_spacing_x, pixel_spacing_y, manufacturer, manufacturer_model_name) or (None, None, None, None).
    """

    try:
        image = Image.open(image_file)
        image.load()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None, None
    
    manufacturer = image.info.get('dcm:Manufacturer')
    manufacturer_model_name = image.info.get("dcm:Manufacturer'sModelName")
    
    spacing_keys = ['dcm:PixelSpacing', 'dcm:ImagerPixelSpacing']

    for key in spacing_keys:

        raw_spacing = image.info.get(key)
        
        if not raw_spacing:
            continue
            
        try:
            parts = raw_spacing.strip().split('\\')
            
            if len(parts) < 2:
                continue
                
            pixel_spacing_y = float(parts[0])  # Row spacing
            pixel_spacing_x = float(parts[1])  # Column spacing
            
            return key, pixel_spacing_x, pixel_spacing_y, manufacturer, manufacturer_model_name
            
        except (IndexError, ValueError) as e:
            continue

    # If no valid key found or parsing failed
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
            spacing_method, info["row_spacing"], info["col_spacing"], info["manufacturer"], info["manufacturer_model_name"] = get_png_pixel_spacing(image_file)

    return info