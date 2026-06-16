# FERDINAND

An open framework for access and AI-ready preprocessing of IMPC 2D radiographs.

## Getting Started

There are two supported ways to run FERDINAnD:

1. Local installation (Python environment on your machine)
2. Docker-based run (recommended)

Recommended for new users: run FERDINAnD with Docker.

1. Check prerequisites:
   - On macOS/Windows: Docker Desktop is running
   - On Ubuntu/Linux: Docker Engine is running
   - `docker compose` is available
2. Create `config/config.yml` from `config/config.example.yml`, the review the following settings:
   - `run_name`
   - `project_root`
   - `data_dir`, `models_dir`, `reports_dir`
   - `angle_detection_model`
   - `docker_group_id`, `jupyter_host`, `jupyter_port`
4. Make sure following directories exist:
   - `data` (for dataset storage)
   - `reports` (for pipeline outputs and reports)
   - `models` (must contained the angle detection model required for image alignment)
3. Start the project:
   - Without Traefik:
   ```bash
   ./docker/compose.sh up -d --build
   ```
   - With Traefik (uses configured hostname in `config.yml`, e.g. `jupyter_host`):
   ```bash
   ./docker/compose.sh --traefik up -d --build
   ```
4. Open JupyterLab:
   - Without Traefik:
   ```text
   http://localhost:8888
   ```
   - With Traefik (uses configured hostname in `config.yml`, e.g. `jupyter_host`):
   ```text
   https://<jupyter_host>/lab
   ```

5. Run notebooks in order:
   - `notebooks/100_fetch_data_from_impc.ipynb`
   - `notebooks/200_download_impc_images.ipynb`
   - `notebooks/300_preprocess_dataset.ipynb`
   - `notebooks/310_curate_dataset.ipynb`
   - `notebooks/320_check_image_quality.ipynb`
   - `notebooks/400_rotate_images.ipynb`

## Purpose

FERDINAnD is an end-to-end framework for building AI-ready datasets from IMPC 2D radiographs. It supports the full workflow from data acquisition to curation and quality control, and can be run either locally or via Docker.

Core capabilities:

- Automated metadata collection from the IMPC Solr image API (`notebooks/100_fetch_data_from_impc.ipynb`).
- Reliable image download and organization for downstream processing (`notebooks/200_download_impc_images.ipynb`, `src/ferdinand/download_utils.py`).
- Center-aware preprocessing and QC for heterogeneous image sources (`notebooks/300_preprocess_dataset.ipynb`, `src/ferdinand/preprocess_and_qc.py`, `src/ferdinand/image_utils.py`).
- Dataset curation and clustering-assisted review (`notebooks/310_curate_dataset.ipynb`, `src/ferdinand/clustering_utils.py`).
- Orientation analysis and image rotation for standardization (`notebooks/400_rotate_images.ipynb`, `src/ferdinand/model_setup.py`, `src/ferdinand/image_utils.py`).
- Reproducible metadata tracking and resume-safe bookkeeping through SQLite (`src/ferdinand/sqlite_utils.py`).

See the `notebooks/` directory for runnable, documented pipeline steps and `src/ferdinand/` for the programmatic APIs used by the notebooks.

## Prerequisites

- Docker path:
  - macOS/Windows: Docker Desktop
  - Ubuntu/Linux: Docker Engine
  - Docker Compose v2 (`docker compose`)
- Local Python path:
  - Python 3.10
  - `pip`
  - virtual environment (`venv` or conda)

## Run with Docker Compose

Build and start the project (JupyterLab on port `8888`):

- Without Traefik:
```bash
./docker/compose.sh up -d --build
```
- With Traefik (uses configured hostname in `config.yml`, e.g. `jupyter_host`):
```bash
./docker/compose.sh --traefik up -d --build
```

Open JupyterLab in your browser:

- Without Traefik:
```text
http://localhost:8888
```
- With Traefik (uses configured hostname in `config.yml`, e.g. `jupyter_host`):
```text
https://<jupyter_host>/lab
```

Show logs:

```bash
./docker/compose.sh logs -f
```

Open a shell in the running container:

```bash
./docker/compose.sh exec ferdinand bash
```

Stop the project:

```bash
./docker/compose.sh down
```

### Config-based Mounts

The compose mounts for data, models and reports are generated from `config/config.yml`.

The helper script reads `data_dir`, `models_dir` and `reports_dir` from `config/config.yml`, writes `.env.compose`, and runs:

```bash
docker compose -f docker/compose.yml --env-file docker/.env.compose ...
```

To only refresh the env file:

```bash
./docker/generate_compose_env.py
```

## Installation (Local, Ubuntu/MacOS)

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

## Troubleshooting

### `img_to_array` ImportError

Depending on the TensorFlow/Keras version in your environment, `img_to_array` may live in different modules.
Use a compatibility import in `src/ferdinand/image_utils.py`:

```python
try:
    from tensorflow.keras.utils import img_to_array
except ImportError:
    try:
        from tensorflow.keras.preprocessing.image import img_to_array
    except ImportError:
        from keras.utils import img_to_array
```

After changing imports, rebuild the container image:

```bash
./docker/compose.sh up -d --build
```

### `403: Blocking request from unknown origin`

If `/files/...` URLs return 403, make sure the container is started via `./docker/compose.sh` and includes:

- `--ServerApp.root_dir=/app`
- `--ServerApp.allow_origin=*`
- `--ServerApp.allow_remote_access=True`

Then restart:

```bash
./docker/compose.sh down
./docker/compose.sh up -d --build
```

### `/lab/tree/...` vs `/files/...`

- `/lab/tree/...` opens the JupyterLab file browser view.
- `/files/...` serves files directly (needed for HTML `<img src>` links).

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

Holger Maier, Ralph Steinkamp & Elida Schneltzer, Experimental Genetics, Helmholtz Munich
and contributors. See [GitHub contributors](https://github.com/ExperimentalGenetics/FERDINAnD/graphs/contributors)
