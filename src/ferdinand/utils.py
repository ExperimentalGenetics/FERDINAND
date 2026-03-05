
import logging
import os
import math
import cv2

import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

from ferdinand import sqlite_utils as sqlutl

"""
This module provides utility functions for handling image data, including logging setup, building standardized directory paths, selecting images from a database, plotting image grids, 
creating thumbnails, and generating HTML galleries. These functions are designed to facilitate the management and visualization of image datasets in a consistent"""

def setup_logger(log_file, log_level=logging.DEBUG):
    """
    Set up a logger that writes to both a file and the console.
    :param log_file: Path to the log file.
    :param log_level: Logging level (default: DEBUG).
    :return: configured logger."""
    # create a logger with the module's name
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # clear old handlers if the logger already has any
    if logger.hasHandlers():
        logger.handlers.clear()

    # file handler: write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # console handler: print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # add both handlers to the logger
    logger.addHandler(file_handler)
    # if log_level <= logging.DEBUG:
    #     logger.addHandler(console_handler)

    return logger

def build_local_image_dir(dir_path, center, cohort_type, gene, mouse_sex):
    """
    Builds a standardized directory path for storing images based on the provided parameters.
    :param dir_path: Base directory path where images are stored.
    :param center: Name of the center (e.g., "MPI-CBG").
    :param cohort_type: Type of cohort (e.g., "mutant" or "control").
    :param gene: Gene symbol (required if cohort_type is "mutant").
    :param mouse_sex: Sex of the mouse (e.g., "male" or "female").
    :return: Constructed directory path as a string.
    """
    # validate inputs
    if not dir_path:
        raise ValueError("dir_path cannot be empty")
    
    required_params = {'center': center, 'cohort_type': cohort_type, 'mouse_sex': mouse_sex}
    for param_name, param_value in required_params.items():
        if not param_value:
            raise ValueError(f"{param_name} cannot be empty or None")

    # gene is required for mutants
    if cohort_type.lower() == 'mutant' and not gene:
        raise ValueError("gene cannot be empty for mutant cohort_type")

    try: 
        # normalize strings: strip whitespace and replace spaces with underscores
        center_normalized = str(center).strip().replace(' ', '_')
        cohort_normalized = str(cohort_type).strip()
        sex_normalized = str(mouse_sex).strip()
        
        # build path components
        path_components = [str(dir_path), center_normalized, cohort_normalized]
        
        # add gene only for mutants
        if cohort_type.lower() == 'mutant':
            gene_normalized = str(gene).strip().replace(' ', '_')
            path_components.append(gene_normalized)
        
        # add sex
        path_components.append(sex_normalized)
        
        # build final path
        final_path = os.path.join(*path_components)
        
        return final_path
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid parameter types: {e}")

def select_random_jpegs(conn, db_table, no_of_images, image_path, center=None):
    """
    Selects a specified number of random images from the database and returns their file paths.
    :param conn: Active sqlite connection.
    :param db_table: Name of the database table to query.
    :param no_of_images: Number of random images to select.
    :param image_path: Base path to the images.
    :param center: Center to filter images by (optional).
    :return: List of file paths to the selected images.
    """  
    df_random_rows = sqlutl.select_random_rows_with_images(conn, db_table, no_of_images, center)
    selected_images = []
    for idx, row in df_random_rows.iterrows():
        img_path = os.path.join(build_local_image_dir(image_path, 
                                                      row['center'], 
                                                      row['cohort_type'], 
                                                      row['gene_symbol'], 
                                                      row['sex']), f"{row['omero_id']}.jpg")
        selected_images.append(img_path)
    
    return selected_images

def select_jpegs(conn, db_table, image_path, center=None):
    """
    Selects images from the database and returns their file paths.
    :param conn: Active sqlite connection.
    :param db_table: Name of the database table to query.
    :param image_path: Base path to the images.
    :param center: Center to filter images by (optional).
    :return: List of file paths to the selected images.
    """
    df_rows = sqlutl.select_rows(conn=conn, db_table=db_table, center=center)
    selected_images = []
    for index, row in df_rows.iterrows():
        img_path = os.path.join(build_local_image_dir(image_path, 
                                                      row['center'], 
                                                      row['cohort_type'], 
                                                      row['gene_symbol'], 
                                                      row['sex']), f"{row['omero_id']}.jpg")
        selected_images.append(img_path)
    
    return selected_images

def select_jpegs_by_column(conn, db_table, column, value, source_path):
    """
    Selects images from the database where a specific column matches a given value and returns their file paths.
    :param conn: Active sqlite connection.
    :param db_table: Name of the database table to query.
    :param column: Column name to filter by.
    :param value: Value to match in the specified column.
    :param source_path: Base path to the images.
    :return: List of file paths to the selected images.
    """
    return select_jpegs_by_columns(conn, db_table, filters={column: value}, source_path=source_path)

def select_jpegs_by_columns(conn, db_table, filters: dict, source_path): 
    """
    Selects images from the database where specific columns match given values and returns their file paths.
    :param conn: Active sqlite connection.
    :param db_table: Name of the database table to query.
    :param filters: Dictionary of column-value pairs to filter by.
    :param source_path: Base path to the images.
    :return: List of file paths to the selected images.
    """  
    try: 
        df_rows = sqlutl.select_rows_by_columns(conn=conn, 
                                                db_table=db_table,
                                                filters=filters)
    except ValueError as e: 
        print(f"{e}\n")
        return []
    
    selected_images = []
    for index, row in df_rows.iterrows():
        img_path = os.path.join(build_local_image_dir(source_path, 
                                                      row['center'], 
                                                      row['cohort_type'], 
                                                      row['gene_symbol'], 
                                                      row['sex']), f"{row['omero_id']}.jpg")
        selected_images.append(img_path)
    
    return selected_images

def plot_random_image_grid(conn, db_table, image_path, no_of_images, centers, 
                           cols=3, cell_size=2.5):
    """
    Plots a grid of random images from the specified centers.
    :param conn: Active sqlite connection.
    :param db_table: Name of the database table to query.
    :param image_path: Base path to the images.
    :param no_of_images: Number of random images to select and plot for each center.
    :param centers: List of centers to select images from.
    :param cols: Number of columns in the grid (default: 3).
    :param cell_size: Size of each cell in the grid (default: 2.5).
    :return: Dictionary mapping each center to the list of selected image file paths.
    """
    rows = math.ceil(no_of_images / cols)

    # compute figure and font size based on cell size
    figsize = (cols * cell_size, rows * cell_size)
    fontsize = max(cell_size * 4, 6)  # minimum font size = 6

    fig_fontsize = max(figsize[1] * 1.5, 12)  # proportional to figure height
    # fig_fontsize = 14

    rnd_selected_images = {}

    for center in centers:

        # Replace spaces with underscores in center names for path formatting
        center_stripped = center.replace(' ', '_')

        print(f" processing images from: {os.path.join(image_path, center_stripped)}")

        # Select random images from the path
        rnd_selected_images[center] = select_random_jpegs(conn, db_table, no_of_images, image_path, 
                                                          center)

        if rnd_selected_images[center] is None or len(rnd_selected_images[center]) == 0:
            print(f" WARNING: No images found for center {center}!")
            continue

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f"Images from {center}", fontsize=fig_fontsize)

        for i, ax in enumerate(axes.flat):
            if i < no_of_images:
                # Load the image in grayscale mode
                image = cv2.imread(rnd_selected_images[center][i], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"FileNotFoundError: Failed to load image {rnd_selected_images[center][i]}!")
                    ax.remove()
                    continue

                ax.imshow(image, cmap="gray")  # show image
                title = os.path.basename(rnd_selected_images[center][i])
                ax.set_title(title, fontsize=fontsize)
                ax.axis("off")
            else:
                ax.remove()

        plt.tight_layout()
        # plt.subplots_adjust(top=0.9)
        plt.show()

    return rnd_selected_images

def plot_image_grid(image_files, cols=3, cell_size=2.5, images=None, titles=None):
    """
    Plots a grid of images from the specified file paths or image arrays.
    :param image_files: List of file paths to the images to plot (used if images is None).
    :param cols: Number of columns in the grid (default: 3).
    :param cell_size: Size of each cell in the grid (default: 2.5).
    :param images: List of image arrays to plot (optional, used if image_files is None).
    :param titles: List of titles for each image (optional).
    """
    no_of_images = len(image_files) if images is None else len(images)
    rows = math.ceil(no_of_images / cols)

    # compute figure and font size based on cell size
    figsize = (cols * cell_size, rows * cell_size)
    fontsize = max(cell_size * 4, 6)  # minimum font size = 6

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < no_of_images:
            if images is not None:
                image = images[i]
            else:   
                # Load the image in grayscale mode
                image = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"FileNotFoundError: Failed to load image {image_files[i]}!")
                ax.remove()
                continue

            ax.imshow(image, cmap="gray")  # show image
            title = os.path.basename(image_files[i]) if images is None else f"Image {i+1}" 
            if titles is not None:
                title = f"{title} / {titles[i]}"
            ax.set_title(title, fontsize=fontsize)
            ax.axis("off")
        else:
            ax.remove()

    plt.tight_layout()
    plt.show()

def create_and_save_thumbnails(data: pd.DataFrame, centers: list, 
                               path2images: str, path2thumbnails: str, 
                               size=(64, 64)):
    """
    Creates and saves thumbnail images for the given data and centers.
    :param data: DataFrame containing metadata for the images, including columns 'center', 'cohort_type', 'gene_symbol', 'sex', and 'omero_id'.
    :param centers: List of centers to process.
    :param path2images: Base path to the original images.
    :param path2thumbnails: Base path to save the thumbnail images.
    :param size: Size of the thumbnail images (default: (64, 64)).
    """
    for center in centers:
        df_center = data[data.center == center]

        for row in df_center.itertuples(index=False):
            input_dir = build_local_image_dir(
                path2images,
                row.center,
                row.cohort_type,
                row.gene_symbol,
                row.sex,
            )
            input_path = os.path.join(input_dir, f"{row.omero_id}.jpg")

            output_dir = build_local_image_dir(
                path2thumbnails,
                row.center,
                row.cohort_type,
                row.gene_symbol,
                row.sex,
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{row.omero_id}.jpg")

            # skip if thumbnail already exists
            if os.path.exists(output_path):
                continue

            try:
                with Image.open(input_path) as img:
                    img.thumbnail(size)
                    img.save(output_path, "JPEG")
            except FileNotFoundError:
                print(f"Missing image: {input_path}")
            except OSError as e:
                print(f"Cannot process {input_path}: {e}")

def create_thumbnail_gallery(data: pd.DataFrame, 
                             html_file_name: str, 
                             title: str, 
                             html_dir: str, 
                             thumbnails_dir: str,
                             source_dir: str, 
                             alignment_details: bool = False):
    """
    Creates an HTML gallery of thumbnail images organized by center, with optional alignment details.
    :param data: DataFrame containing metadata for the images, including columns 'center', 'cohort_type', 'gene_symbol', 'sex', 'omero_id', and optionally 'fst_angle_prediction' and 'snd_angle_prediction' for alignment details.
    :param html_file_name: Name of the HTML file to create 
    :param title: Title to display at the top of the gallery.
    :param html_dir: Directory where the HTML file will be saved.
    :param thumbnails_dir: Base directory where thumbnail images are stored.
    :param source_dir: Base directory where original images are stored (used for linking).
    :param alignment_details: Whether to include alignment details in the gallery (default: False).
    """
    html_file = os.path.join(html_dir, html_file_name)
    print(f"Creating thumbnail gallery at: {html_file}")

    # path relative to the thumbnails directory
    # source_dir = os.path.join('..', os.path.basename(source_dir))

    with open(html_file, "w") as f:
        f.write(f"<html><body><h2>{title}</h2>")
        for center in data['center'].unique():
            _df = data[data['center'] == center].copy()
            total = len(_df)
            if alignment_details:
                _df.sort_values(by='snd_angle_prediction', inplace=True)
                aligned = len(_df[(_df['snd_angle_prediction'] < 10) | (_df['snd_angle_prediction'] > 350)])
                f.write(f"<h3>Center: {center} ({int((aligned/total)*100) if total > 0 else 0}% aligned)</h3></br>")
            else:
                f.write(f"<h3>Center: {center} ({total} images) </h3></br>")
             
            for idx, row in _df.iterrows():
                orig_file = os.path.join(build_local_image_dir(source_dir, row['center'], row['cohort_type'], row['gene_symbol'], row['sex']), 
                                         f"{row['omero_id']}.jpg")
                orig_file = os.path.relpath(orig_file, html_dir)
                thumb_file = os.path.join(build_local_image_dir(thumbnails_dir, row['center'], row['cohort_type'], row['gene_symbol'], row['sex']), 
                                          f"{row['omero_id']}.jpg")
                thumb_file = os.path.relpath(thumb_file, html_dir)
                # f.write(f'<a href="{orig_file}"><img src="{thumb_file}" width="128" ></a> ')
                if alignment_details:
                    f.write(f'''
                            <div style="display:inline-block; text-align:center;">
                            <a href="{orig_file}"><img src="{thumb_file}" width="128"></a><br>
                            {row['omero_id']} ({row['fst_angle_prediction']}°/{row['snd_angle_prediction']}°)
                            </div>''')
                else:
                    f.write(f'''
                            <div style="display:inline-block; text-align:center;">
                            <a href="{orig_file}"><img src="{thumb_file}" width="128"></a><br>
                            {row['omero_id']}
                            </div>''')
        f.write("</body></html>")
