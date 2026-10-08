import logging
import os
import requests

import pandas as pd

from types import SimpleNamespace
from pathlib import Path

from ferdinand.image_utils import detect_file_format, get_image_info

"""
Utilities for downloading IMPC metadata tables and image assets.

The module supports three main tasks:
- downloading raw metadata CSV files from the IMPC API for selected centers
  and parameters,
- merging those raw files into center-level and global analysis tables, and
- downloading OMERO-backed image files while extracting image metadata.

Most helpers write files to disk as part of their normal workflow, so callers
should provide directory paths that already exist and are writable.
"""

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

def fetch_metadata(mode: str, my_centers: list, 
                   my_parameters: pd.DataFrame, expected_df: pd.DataFrame, dirs: SimpleNamespace, 
                   api_base_url: str, api_fields: str, api_options: str, logger):
    """
    Download raw IMPC metadata CSV files for selected parameters and centers.

    Parameters
    ----------
    mode : str
        Download mode. `"TEST"` limits each request to 20 rows; any other
        value requests approximately the expected number of rows plus a small
        buffer.
    my_centers : list[str]
        Phenotyping centers to request from the API.
    my_parameters : pandas.DataFrame
        Parameter table containing at least a `stable_id` column.
    expected_df : pandas.DataFrame
        Lookup table of expected row counts indexed by parameter stable ID and
        with one column per center.
    dirs : types.SimpleNamespace
        Namespace containing `raw_data_dir`, where the downloaded CSV files
        will be written.
    api_base_url : str
        Base IMPC API URL prefix before parameter and filter fragments are
        appended.
    api_fields : str
        Query-string fragment listing requested API fields.
    api_options : str
        Additional query-string fragment appended to every request.
    logger : logging.Logger
        Logger used for progress and warning output.

    Returns
    -------
    None
        The function writes one raw CSV file per `(parameter, center)` pair and
        logs progress, but does not return a value.

    Raises
    ------
    requests.HTTPError
        Raised when the IMPC API responds with a non-success status code.
    """
    # loop over all parameters and centers and download the metadata
    number_centers = len(my_centers)
    for my_stable_id in my_parameters.stable_id:
 
        logger.info(f"\n==================================================================================================================================")
        logger.info(f"DOWNLOADING: {my_stable_id} from {str(number_centers)} centers") 
        logger.info(f"==================================================================================================================================")

        counter = 0
    
        # loop over all phenotyping centers
        for my_center in my_centers:
        
            if my_center not in expected_df.columns:
                continue

            counter = counter + 1

            # build file name
            my_center_stripped = my_center.replace(" ", "_")
            my_file = os.path.join(dirs.raw_data_dir, f"parameter-{my_stable_id}-{my_center_stripped}-raw_data.csv")
            # build center filter
            my_center_quoted   = '\"' + my_center + '\"'
            my_center_filter = "&fq=phenotyping_center:" + my_center_quoted

            # get expected number of data points
            expected_data_points = expected_df[my_center][my_stable_id]

            if mode=="TEST":
                # in TEST mode, only download 20 rows of data
                api_lines = '&rows=20'
            else:
                # dynamically set number of rows to download (expected + 100)
                api_lines  = '&rows=' + str(expected_data_points + 10)

            # build the URL from the components defined above
            my_url = api_base_url + my_stable_id + my_center_filter + api_fields + api_lines + api_options
        
            logger.setLevel(logging.INFO)
            logger.info(f"\n{str(counter)}. trying to download expected {expected_data_points} data points of {my_stable_id} from {my_center} using:\n{my_url}")

            # replace whitespaces, e.g. in "MRC Harwell" and "UC Davis"
            my_url_quoted = my_url.replace(" ", "%20")
        
            print(my_url_quoted)
            response = requests.get(my_url_quoted)
            response.raise_for_status()  # Fehler werfen, falls HTTP != 200

            outfile = Path(my_file)
            outfile.write_bytes(response.content)

            # os.system(command_string)
            with open(my_file) as f:
                row_count = sum(1 for line in f)

            # ERROR: less data than expected
            if row_count < expected_data_points: 
                logger.error(f"\n############ WARNING: {expected_data_points} expected, but only {row_count - 1} downloaded! ############")

        # error or not: inform
        logger.info(f"\ndownloaded {row_count - 1} rows of data and stored in: {my_file}\n-----------\n")

def merge_metadata_files(config: dict, dirs: SimpleNamespace, logger: logging.Logger) -> pd.DataFrame:
    """
    Merge raw metadata CSV files into center-specific and global tables.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing at least:
        - `centers`: list of center names,
        - `parameters`: records with `stable_id`, `description`, `week`, and
          `work_name`,
        - `sqlite_db`: basename used for the all-centers CSV output.
        Optional keys such as `max_pipeline_deviation`, `mouse_columns`, and
        `additional_api_fields` further control filtering and column layout.
    dirs : types.SimpleNamespace
        Namespace containing `raw_data_dir`, `center_data_dir`, and
        `all_data_dir`.
    logger : logging.Logger
        Logger used for processing, filtering, and merge progress.

    Returns
    -------
    pandas.DataFrame
        Combined dataframe across all configured centers after filtering,
        normalization, and per-center merges.

    Notes
    -----
    This function has significant side effects. It writes one merged CSV per
    center into `dirs.center_data_dir` and one all-centers CSV into
    `dirs.all_data_dir`.

    The raw input files are expected to include IMPC columns such as
    `external_sample_id`, `omero_id`, `age_in_weeks`, `phenotyping_center`,
    `date_of_birth`, and several pipeline/procedure metadata fields that are
    later dropped or renamed.
    """

    # read config parameters
    my_centers = config['centers']
    my_parameters = pd.DataFrame(config['parameters'])
    my_max_pipeline_deviation = config.get('max_pipeline_deviation', None)
    my_mouse_columns = config.get('mouse_columns', [
        'mouse_id', 'external_sample_id',  'gene_symbol', 'sex', 'cohort_type', 'strain', 'center', 'dob'
    ])
    additional_api_fields = config.get('additional_api_fields', [])
    
    # some counters to inform about the progress
    number_centers = len(my_centers)
    number_params  = len(my_parameters)
    center_counter = 0
    param_counter  = 0

    # list to collect all center dataframes
    all_df_list = []

    print(f"reading raw data from: {dirs.raw_data_dir} ")

    # loop over all phenotyping centers
    for my_center in my_centers:
        my_center_stripped = my_center.replace(" ", "_")

        center_counter = center_counter + 1

        print(f"\n==================================================================================================================================")
        print(f"Center {center_counter} of {number_centers}: {my_center}")
        logger.info(f"\n==================================================================================================================================")
        logger.info(f"PROCESSING CENTER {center_counter} OF {number_centers}: {my_center} - checking for {number_params} parameter files") 
        logger.info(f"==================================================================================================================================")

        # dataframe to collect data from all parameters of current center
        combined_df = pd.DataFrame()

        # loop over all parameters
        for idx in my_parameters.index:
            parameter_id = my_parameters.at[idx, "stable_id"]
            description  = my_parameters.at[idx, "description"]
            due_week     = int(my_parameters.at[idx, "week"])
            work_name    = my_parameters.at[idx, "work_name"]

            param_counter = param_counter + 1

            print(f"   Parameter {param_counter} of {number_params}: {parameter_id} ({work_name})")
            logger.info(f"Parameter {param_counter} of {number_params}: {parameter_id} ({work_name}) ")
        
            # build expected file name for raw data of current center and current parameter
            my_file = os.path.join(dirs.raw_data_dir, f"parameter-{parameter_id}-{my_center_stripped}-raw_data.csv") 

            # check if expected file exists: if yes - process it
            if os.path.exists(my_file):
           
                logger.info(f"processing \"./{my_file}\"... ")
                df = pd.read_csv(my_file) 

                # inform
                logger.info(f"   {str(len(df))} rows in file")
                logger.info(f"   distinct values in columns, sorted - age week should be {str(due_week)}")
                for col in ["sex", "age_in_weeks", "pipeline_stable_id", "life_stage_name", "biological_sample_group", "strain_name"]:
                    unique_vals = df[col].dropna().unique()
                    unique_vals_str = sorted([str(x) for x in unique_vals])
                    logger.info(f"      {col}: {unique_vals_str}")
                logger.info(f"   distinct entries")
            
                for col in ["external_sample_id", "gene_symbol"]:
                    logger.info(f"      {col}: {len(df[col].unique())}")

                #----------------------------------------------------
                # duplicates filter
                duplicates = df[df.duplicated(subset=['external_sample_id', 'omero_id'], keep=False)]  # get duplicates
                logger.info(f"   duplicates (same mouse & same value) removed: {str(len(duplicates.external_sample_id.unique()))}")
                df = df.drop_duplicates(subset=['external_sample_id', 'omero_id'], keep='first')       # remove them, keep first
                #logger.info(str(len(duplicates.external_sample_id.unique())))                         # inform

                # handle age week
                weeks = df["age_in_weeks"].value_counts().to_string()
                weeks_str = str(weeks)
                logger.info(f"   " + weeks_str.replace('\n', '\n      '))

                #----------------------------------------------------
                # pipeline filter: remove all entries that deviate from the due_week according to the pipeline
                initial_length = len(df)
                if my_max_pipeline_deviation:
                    df_filtered = df[(df['age_in_weeks'] >= due_week - my_max_pipeline_deviation) & 
                                    (df['age_in_weeks'] <= due_week + my_max_pipeline_deviation)
                    ]

                    removed_rows = initial_length - len(df_filtered)

                    # apply changes and report
                    df = df_filtered

                    if removed_rows > 0:
                        print(f"       removed {removed_rows} mice, since measure week deviated more than {my_max_pipeline_deviation} weeks from due week given by pipeline!")
                        logger.info(f"      removed {removed_rows} mice, since measure week deviated more than {my_max_pipeline_deviation} weeks from due week given by pipeline!")
                #----------------------------------------------------
            
                # rebuild dataframe
                # 1) drop columns
                df = df.drop(columns=['pipeline_name', 'pipeline_stable_id','life_stage_name','procedure_name', 'procedure_stable_id'])
                df = df.drop(columns=['age_in_days', 'parameter_stable_id', 'parameter_name', 'age_in_weeks'])
            
                # 2) rename columns
                df = df.rename(columns={'phenotyping_center'      : 'center',
                                        'date_of_birth'           : 'dob',
                                        'biological_sample_group' : 'cohort_type',
                                        'strain_name'             : 'strain', 
                                        'file_type'               : 'impc_file_type'
                             })
                df["mouse_id"] = df["center"].astype(str) + "_" + df["external_sample_id"].astype(str)
            
                # 3) sort columns
                if additional_api_fields:
                    df = df[list(my_mouse_columns) + ['impc_file_type', 'omero_id', 'date_of_experiment'] + list(additional_api_fields)]
                else:
                    df = df[list(my_mouse_columns) + ['impc_file_type', 'omero_id', 'date_of_experiment']]
                for col in ['parameter_association_name', 'parameter_association_stable_id', 'parameter_association_value']:
                    if col not in df.columns:
                        continue
                    else:
                        try:
                            df[col] = df[col].str.split(',', n=1, expand=True)[0]
                        except Exception as e:
                            logger.info(f"No need to split {col}, not a comma-separated string: {e}")

                # 4) change date format
                df['dob'] = pd.to_datetime(df['dob']).dt.strftime('%Y-%m-%d')
            
                # add new columns
                df['downloaded'] = 'no'
            
                # change the values in cohort_type
                df['cohort_type'] = df['cohort_type'].replace({'experimental': 'mutant'})

                #----------------------------------------------------
                # now we remove all entries where date of birth (dob) before "2010-11-15"
                initial_length = len(df)
                df_filtered = df[(df['dob'] >= "2010-11-15")]

                removed_rows = initial_length - len(df_filtered)

                # apply changes and report
                df = df_filtered

                if removed_rows > 0:
                    print(f"       removed {removed_rows} mice born before 2011 (non-IMPC mice)")
                    logger.info(f"      removed {removed_rows} mice born before 2011 (non-IMPC mice)")
                #---------------
            
                # merge df to combined_df to create a center-specific combined dataframe
                # merge is based on key_columns
                key_columns = my_mouse_columns
            
                if combined_df.empty:
                    combined_df = df
                else:
                    logger.info(f"   merging data ...")
                    combined_df = pd.merge(combined_df, df, on=key_columns, how='outer')
        
            # in case no file for current parameter found
            else:
                print(f"   file {my_file} NOT FOUND!")
                logger.info(f"file {my_file} ===========> NOT FOUND! ")
            print(" ")

            # done with current parameter

        # done with current center
  
        logger.info(f"total rows (after filter): {len(combined_df)}")

        # ------------------------------------------------    

        # add center dataframe to list for overall IMPC dataframe
        all_df_list.append(combined_df)
    
        # save combined_df to file
        out_filename = os.path.join(dirs.center_data_dir, f"IMPC-{my_center_stripped}-data.csv")
        print(f"saving combined data for center \"{my_center}\" to {out_filename}")
        logger.info(f"\nsaving combined data for center \"{my_center}\" to {out_filename}")
        combined_df.to_csv(out_filename, index=False)

        # reset parameter counter, be ready for next center
        param_counter = 0     

    # done with all centers, now create overall IMPC dataframe
    all_df = pd.concat(all_df_list, ignore_index=True)        # concat all center dataframes
    all_df['mouse_id'] = all_df['mouse_id'].astype(str)       # convert mouse_ids to string
    all_df = all_df.sort_values(by='mouse_id')                # sort by mouse_id
    all_df = all_df.reset_index(drop=True)                    # re-index

    # create logger
    logger = logging.getLogger('all')
    logger.setLevel(logging.INFO)

    # save overall IMPC media files data csv file
    out_filename = os.path.join(dirs.all_data_dir, f"{config['sqlite_db']}.csv")
    print(f"\n----------\nsaving combined data for all centers to {out_filename}")
    logger.info(f"\nsaving combined data for all centers {out_filename}")
    all_df.to_csv(out_filename, index=False)

    logger.info(f"Total: {len(all_df)-1} mice")
    logger.info(f"Total: {all_df['gene_symbol'].nunique()} genes")

    # copy dataframe (we need it later to merge it with weights dataframe
    all_data_df = all_df.copy()

    return all_data_df

def get_extension(file_type: str | None) -> str:
    """
    Map a detected image format name to a filename extension.

    Parameters
    ----------
    file_type : str | None
        File-type label such as `"JPEG"` or `"DICOM"`. Lookup is
        case-insensitive. A missing value falls back to the generic `"bin"`
        extension.

    Returns
    -------
    str
        Preferred extension without a leading dot. Unknown formats fall back to
        `"bin"`.
    """
    if file_type is None:
        return "bin"
    return IMAGE_FILE_EXTENSION_MAP.get(file_type.upper(), "bin")

def download_image(omero_id, local_image_path, is_jpeg=True, save_local=True, logger=None):
    """
    Download an IMPC image by OMERO ID and extract image metadata.

    Parameters
    ----------
    omero_id : str | int
        OMERO image identifier appended to the IMPC image endpoint.
    local_image_path : str | os.PathLike
        Directory where the downloaded image should be saved if `save_local` is
        `True`.
    is_jpeg : bool, optional
        If `True`, request the rendered JPEG endpoint. If `False`, request the
        original archived file endpoint.
    save_local : bool, optional
        Whether to write the downloaded bytes to `local_image_path`.
    logger : logging.Logger, optional
        Logger used for debug output.

    Returns
    -------
    pandas.DataFrame
        Single-row dataframe describing the image. JPEG downloads return a
        JPEG-specific schema with columns such as `jpeg_width` and
        `jpeg_height`; non-JPEG downloads keep the original dimension/spacing
        columns and add `file_extension`. Both variants include `omero_id`.

    Raises
    ------
    RuntimeError
        Raised when the image request returns a non-200 status code.

    Notes
    -----
    The filename extension is inferred from the HTTP response rather than from
    `is_jpeg` alone.
    """
    
    # construct URL based on image format (JPEG or original)
    image_url = os.path.join(f"{IMPC_JPEG_URL if is_jpeg else IMPC_ORIGINAL_URL}", str(omero_id))
    print(f"Downloading image from {image_url} ...")

    # request image from IMPC server
    response = requests.get(image_url, headers=HEADERS)

    # check if request was successful
    if response.status_code == 200:
        # detect file format from response headers/content
        file_type = detect_file_format(response)
        file_ext = get_extension(file_type)

        # construct local file path
        local_image_file = os.path.join(local_image_path, f"{omero_id}.{file_ext}")
        
        # extract image metadata (width, height, spacing, DPI, etc.)
        image_info = get_image_info(local_image_file, response)
        image_info = pd.DataFrame(image_info, index=[0])
        
        # optionally save image file to local disk
        if save_local:
            with open(local_image_file, 'wb') as f:
                f.write(response.content)
  
        # process metadata based on image format
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
            # for DICOM: remove file path and DPI, keep dimension/spacing info
            drop_cols = [c for c in ["file", "dpi"] if c in image_info.columns]
            image_info = image_info.drop(columns=drop_cols)
            image_info['file_extension'] = file_ext

        # add OMERO ID to metadata
        image_info['omero_id'] = omero_id
        
        # log results if logger provided
        if logger is not None:
            for idx, row in image_info.iterrows():
                logger.debug(f"{row}\n")
            logger.debug(f" Download successful: {local_image_file}")
        
        return image_info
    else:
        # raise error if request failed
        raise RuntimeError(f" Request failed for {image_url} with status code {response.status_code}")
