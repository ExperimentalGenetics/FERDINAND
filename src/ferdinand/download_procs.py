import os
import requests

import pandas as pd

from ferdinand.image_procs import detect_file_format, get_image_info

IMPC_ORIGINAL_URL = 'https://www.ebi.ac.uk/mi/media/omero/webgateway/archived_files/download'
IMPC_JPEG_URL = 'https://www.ebi.ac.uk/mi/media/omero/webgateway/render_image'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537'}

IMAGE_FILE_EXTENSION_MAP = {
    "DICOM": "dcm",
    "JPEG": "jpg",
    "PNG": "png",
    "BMP": "bmp",
    "TIFF": "tif"
}

def get_extension(file_type: str) -> str:
    return IMAGE_FILE_EXTENSION_MAP.get(file_type.upper(), "bin")

def download_image(omero_id, local_image_path, is_jpeg=True, save_local=True, logger=None):
    """
    Downloads either JPEG or original DICOM format from IMPC servers, detects file type,
    extracts image metadata (dimensions, spacing), and optionally saves to local disk.
    
    :param omero_id: OMERO image identifier
    :param local_image_path: Local directory path to save image file
    :param is_jpeg: If True, download JPEG version; if False, download original DICOM (default: True)
    :param save_local: If True, save image to disk; if False, only extract metadata (default: True)
    :param logger: optional logger instance for logging results
    :return: pandas.DataFrame with image metadata (dimensions, spacing, file extension, omero_id)
    :raises RuntimeError: If HTTP request fails or server returns error status code
    """
    
    # Construct URL based on image format (JPEG or original)
    image_url = os.path.join(f"{IMPC_JPEG_URL if is_jpeg else IMPC_ORIGINAL_URL}", str(omero_id))

    # Request image from IMPC server
    response = requests.get(image_url, headers=HEADERS)

    # Check if request was successful
    if response.status_code == 200:
        # Detect file format from response headers/content
        file_type = detect_file_format(response)
        file_ext = get_extension(file_type)

        # Construct local file path
        local_image_file = os.path.join(local_image_path, f"{omero_id}.{file_ext}")
        
        # Extract image metadata (width, height, spacing, DPI, etc.)
        image_info = get_image_info(local_image_file, response)
        image_info = pd.DataFrame(image_info, index=[0])
        
        # Optionally save image file to local disk
        if save_local:
            with open(local_image_file, 'wb') as f:
                f.write(response.content)
  
        # Process metadata based on image format
        if is_jpeg:
            # For JPEG: keep only relevant columns and rename with 'jpeg_' prefix
            cols_to_keep = [c for c in ['width', 'height', 'row_spacing', 'col_spacing', 'dpi'] if c in image_info.columns]
            image_info = image_info[cols_to_keep]
            image_info.rename(columns={
                'width': 'jpeg_width',
                'height': 'jpeg_height',
                'col_spacing': 'jpeg_col_spacing',
                'row_spacing': 'jpeg_row_spacing'
            }, inplace=True)
        else:
            # For DICOM: remove file path and DPI, keep dimension/spacing info
            drop_cols = [c for c in ["file", "dpi"] if c in image_info.columns]
            image_info = image_info.drop(columns=drop_cols)
            image_info['file_extension'] = file_ext

        # Add OMERO ID to metadata
        image_info['omero_id'] = omero_id
        
        # Log results if logger provided
        if logger is not None:
            for idx, row in image_info.iterrows():
                logger.debug(f"{row}\n")
            logger.debug(f" Download successful: {local_image_file}")
        
        return image_info
    else:
        # Raise error if request failed
        raise RuntimeError(f" Request failed for {image_url} with status code {response.status_code}")