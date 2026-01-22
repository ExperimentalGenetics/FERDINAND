
import logging
import os

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
    Build a standardized directory path for storing image files.
    :param dir_path: Base directory path
    :param center: Name of the center (spaces replaced by underscores)
    :param cohort_type: Cohort type (e.g., 'mutant', 'control')
    :param gene: Gene name (spaces replaced with underscores, only used for mutants)
    :param mouse_sex: Mouse sex (e.g., 'male', 'female')
    :return: str - Full standardized directory path
    """

    # Validate inputs
    if not dir_path:
        raise ValueError("dir_path cannot be empty")
    
    required_params = {'center': center, 'cohort_type': cohort_type, 'mouse_sex': mouse_sex}
    for param_name, param_value in required_params.items():
        if not param_value:
            raise ValueError(f"{param_name} cannot be empty or None")

    # Gene is required for mutants
    if cohort_type.lower() == 'mutant' and not gene:
        raise ValueError("gene cannot be empty for mutant cohort_type")

    try: 
        # Normalize strings: strip whitespace and replace spaces with underscores
        center_normalized = str(center).strip().replace(' ', '_')
        cohort_normalized = str(cohort_type).strip()
        sex_normalized = str(mouse_sex).strip()
        
        # Build path components
        path_components = [str(dir_path), center_normalized, cohort_normalized]
        
        # Add gene only for mutants
        if cohort_type.lower() == 'mutant':
            gene_normalized = str(gene).strip().replace(' ', '_')
            path_components.append(gene_normalized)
        
        # Add sex
        path_components.append(sex_normalized)
        
        # Build final path
        final_path = os.path.join(*path_components)
        
        return final_path
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid parameter types: {e}")
