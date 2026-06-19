
import logging
import os
import math
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from tqdm import tqdm

from ferdinand import sqlite_utils as sqlutl

"""General utilities for image-path management, plotting, and galleries.

The module provides helpers for configuring logging, building standardized
directory layouts for IMPC images, selecting image file paths from SQLite-backed
metadata, plotting image grids, and generating thumbnails or simple HTML
galleries for inspection.
"""

def setup_logger(log_file, log_level=logging.DEBUG):
    """
    Configure a module logger that writes to a log file.

    Parameters
    ----------
    log_file : str | os.PathLike
        Destination path for the log file.
    log_level : int, optional
        Standard Python logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance scoped to this module.
    """
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
    Build the standardized directory path for a group of local images.

    Parameters
    ----------
    dir_path : str | os.PathLike
        Root directory where images are stored.
    center : str
        Center name, for example `"MPI-CBG"`.
    cohort_type : str
        Cohort type such as `"mutant"` or `"control"`.
    gene : str | None
        Gene symbol. Required when `cohort_type` is `"mutant"`.
    mouse_sex : str
        Mouse sex label.

    Returns
    -------
    str
        Constructed directory path.

    Raises
    ------
    ValueError
        Raised when required inputs are missing or cannot be normalized.
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
    Select random downloaded JPEG paths from the metadata database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Metadata table to query.
    no_of_images : int
        Number of random images to select.
    image_path : str | os.PathLike
        Root directory containing the downloaded images.
    center : str | None, optional
        Optional center filter.

    Returns
    -------
    list[str]
        File paths to the selected JPEG images.
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
    Select all JPEG paths described by the metadata database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Metadata table to query.
    image_path : str | os.PathLike
        Root directory containing the downloaded images.
    center : str | None, optional
        Optional center filter.

    Returns
    -------
    list[str]
        File paths to the selected JPEG images.
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
    Select JPEG paths for rows matching one metadata condition.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Metadata table to query.
    column : str
        Metadata column used as a filter key.
    value : object
        Required value in `column`.
    source_path : str | os.PathLike
        Root directory containing the downloaded images.

    Returns
    -------
    list[str]
        File paths to the selected JPEG images.
    """
    return select_jpegs_by_columns(conn, db_table, filters={column: value}, source_path=source_path)

def select_jpegs_by_columns(conn, db_table, filters: dict, source_path): 
    """
    Select JPEG paths for rows matching multiple metadata conditions.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Metadata table to query.
    filters : dict
        Dictionary of column/value filters passed to
        `sqlite_utils.select_rows_by_columns`.
    source_path : str | os.PathLike
        Root directory containing the downloaded images.

    Returns
    -------
    list[str]
        File paths to the selected JPEG images. Returns an empty list when the
        filter validation fails.
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
    Plot random image grids for one or more centers.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Metadata table to query.
    image_path : str | os.PathLike
        Root directory containing the downloaded images.
    no_of_images : int
        Number of random images to plot per center.
    centers : sequence[str]
        Centers to visualize.
    cols : int, optional
        Number of columns in each grid.
    cell_size : float, optional
        Base figure-cell size used to scale the figure and titles.

    Returns
    -------
    dict[str, list[str]]
        Mapping from center name to the selected image file paths.
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
    Plot a grid of images from file paths or in-memory arrays.

    Parameters
    ----------
    image_files : sequence[str] | None
        Image file paths used when `images` is `None`.
    cols : int, optional
        Number of columns in the grid.
    cell_size : float, optional
        Base figure-cell size used to scale the figure and titles.
    images : sequence[numpy.ndarray] | None, optional
        In-memory images to plot instead of loading from disk.
    titles : sequence[str] | None, optional
        Optional extra titles appended per image.

    Returns
    -------
    None
        The function renders the grid with Matplotlib and does not return a
        value.
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
    Create thumbnail JPEGs for the selected centers.

    Parameters
    ----------
    data : pandas.DataFrame
        Metadata containing at least `center`, `cohort_type`, `gene_symbol`,
        `sex`, and `omero_id`.
    centers : list[str]
        Centers whose images should be processed.
    path2images : str
        Root directory containing the source images.
    path2thumbnails : str
        Root directory where thumbnails should be written.
    size : tuple[int, int], optional
        Maximum thumbnail size passed to `PIL.Image.thumbnail`.

    Returns
    -------
    None
        The function writes thumbnail files to disk and does not return a
        value.
    """
    for center in centers:
        df_center = data[data.center == center]

        pbar = tqdm(df_center.itertuples(index=False), desc="Thumbnails creation")

        for row in pbar:
        # for row in df_center.itertuples(index=False):
            input_dir = build_local_image_dir(
                path2images,
                row.center,
                row.cohort_type,
                row.gene_symbol,
                row.sex,
            )
            input_path = os.path.join(input_dir, f"{row.omero_id}.jpg")

            pbar.set_postfix_str(f"{center}: {input_path}")

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
    Create a simple HTML gallery of thumbnails grouped by center.

    Parameters
    ----------
    data : pandas.DataFrame
        Metadata containing at least `center`, `cohort_type`, `gene_symbol`,
        `sex`, and `omero_id`. When `alignment_details=True`, the dataframe is
        also expected to contain `fst_angle_prediction` and
        `snd_angle_prediction`.
    html_file_name : str
        Output HTML filename.
    title : str
        Title displayed at the top of the gallery.
    html_dir : str
        Directory where the HTML file should be written.
    thumbnails_dir : str
        Root directory containing thumbnail images.
    source_dir : str
        Root directory containing the original images used for the anchor
        targets.
    alignment_details : bool, optional
        Whether to include per-image alignment predictions and center-level
        alignment summaries.

    Returns
    -------
    None
        The function writes an HTML gallery to disk and does not return a
        value.
    """
    html_dir_abs = os.path.abspath(html_dir)
    thumbnails_dir_abs = os.path.abspath(thumbnails_dir)
    source_dir_abs = os.path.abspath(source_dir)
    os.makedirs(html_dir_abs, exist_ok=True)

    html_file = os.path.join(html_dir_abs, html_file_name)
    print(f"Creating thumbnail gallery at: {html_file}")

    def _href_from_html(target_file: str) -> str:
        # Build browser-friendly relative links from html location to file location.
        return Path(os.path.relpath(os.path.abspath(target_file), html_dir_abs)).as_posix()

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
                orig_file = os.path.join(
                    build_local_image_dir(source_dir_abs, row['center'], row['cohort_type'], row['gene_symbol'], row['sex']),
                    f"{row['omero_id']}.jpg"
                )
                thumb_file = os.path.join(
                    build_local_image_dir(thumbnails_dir_abs, row['center'], row['cohort_type'], row['gene_symbol'], row['sex']),
                    f"{row['omero_id']}.jpg"
                )
                orig_href = _href_from_html(orig_file)
                thumb_href = _href_from_html(thumb_file)
                # f.write(f'<a href="{orig_file}"><img src="{thumb_file}" width="128" ></a> ')
                if alignment_details:
                    f.write(f'''
                            <div style="display:inline-block; text-align:center;">
                            <a href="{orig_href}"><img src="{thumb_href}" width="128"></a><br>
                            {row['omero_id']} ({row['fst_angle_prediction']}°/{row['snd_angle_prediction']}°)
                            </div>''')
                else:
                    f.write(f'''
                            <div style="display:inline-block; text-align:center;">
                            <a href="{orig_href}"><img src="{thumb_href}" width="128"></a><br>
                            {row['omero_id']}
                            </div>''')
        f.write("</body></html>")
