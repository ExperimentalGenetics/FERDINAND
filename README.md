# FERDINAnD

An open framework for access and AI-ready preprocessing of IMPC 2D radiographs.

## Purpose

FERDINAnD provides an end-to-end pipeline to fetch, curate, preprocess and prepare 2D radiograph image data from the IMPC for downstream AI workflows. It focuses on:

- Automated data acquisition (IMPC Solr image API) and per-center CSV collection (`notebooks/100_fetch_data_from_impc.ipynb`).
- Robust image download and metadata extraction (`notebooks/020_download_impc_radiographs.ipynb`, `src/ferdinand/download_utils.py`).
- Per-center, center-specific preprocessing tuned for different data sources and QC (`notebooks/300_preprocess_dataset.ipynb`, `src/ferdinand/preprocess_and_qc.py`, `src/ferdinand/image_utils.py`).
- Light-weight metadata storage and bookkeeping via SQLite (`src/ferdinand/sqlite_procs.py`) so downstream steps can resume safely.

See the `notebooks/` directory for runnable, documented pipeline steps and `src/ferdinand/` for the programmatic APIs used by the notebooks.

## Installation (Ubuntu/MacOS)

It is recommended to install the project inside a __virtual environment__. To do this, follow these steps for Python 3.10:

1. __Clone the repository__, then change to its directory:

    ```bash
    git clone https://github.com/ExperimentalGenetics/FERDINAnD.git
    cd FERDINAnD
    ```

2. __Create and activate a virtual environment__ (example with Python's built-in venv):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. __Upgrade __pip____. Optional, but recommended:

    ```bash
    pip install --upgrade pip
    ```

4. __Select the appropriate requirements file__ for your operating system (e.g., on macOS M1-architecture: ```ln -s requirements-arm.txt requirements.txt```).
5. __Install the dependencies__:

    ```bash
    pip install setuptools
    pip install -e .
    ```

To __deactivate__ the environment:

```bash
deactivate
```

---

## Quick Start Example

1. Fetch metadata and images from IMPC:
    - Run the notebook: `notebooks/100_fetch_data_from_impc.ipynb`
    - Download images: `notebooks/200_download_impc_images.ipynb`
2. Preprocess and curate dataset:
    - Preprocessing: `notebooks/300_preprocess_dataset.ipynb`
    - Curation: `notebooks/310_curate_dataset.ipynb`
3. Check image quality and rotate images (optional):
    - `notebooks/320_check_image_quality.ipynb`
    - `notebooks/400_rotate_images.ipynb`

All steps are modular and can be run independently. See the `notebooks/` directory for detailed documentation and example usage.

---

## Project Structure

- `src/ferdinand/`: Core Python modules (data download, preprocessing, clustering, utilities)
- `notebooks/`: Jupyter notebooks for each pipeline step
- `data/`: Project data (raw, processed, images, thumbnails)
- `models/`: Trained models
- `reports/`: HTML reports, logs, cluster galleries
- `config/`: Configuration files (YAML)

---

## Main Modules

- `download_utils.py`: Download metadata and images from IMPC
- `preprocess_and_qc.py`: Preprocess images, quality control
- `clustering_utils.py`: Feature extraction, clustering (Leiden, KNN, etc.)
- `sqlite_utils.py`: Lightweight metadata storage and queries
- `image_utils.py`: Image manipulation and augmentation

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

Elida Schneltzer, Experimental Genetics, Helmholtz Munich
and contributors. See [GitHub contributors](https://github.com/ExperimentalGenetics/FERDINAnD/graphs/contributors)
