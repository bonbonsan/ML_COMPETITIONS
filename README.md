# ML_COMPETITIONS

This repository is a custom-built machine learning library developed in Python 3.11.  
It is designed for competitive data science tasks (e.g., Kaggle) and provides a modular and reusable codebase with a unified interface.

## Module Descriptions

- `documents/` – Cheat sheets and slides obtained from online sources
- `my_library/` – Core machine learning library  
  - `configs/` – Configuration classes for each algorithm using dataclasses  
  - `data/` – Sample datasets collected from public sources  
    - `samples/` – Public sample datasets (included in Git and Docker)  
    - Other folders – Private or competition datasets (excluded from Git, but shared with Docker)  
  - `ensembles/` – Classes for combining predictions (ensembling)  
  - `feature_engineerings/` – Functions for feature engineering  
  - `logs/` – Output destination for log files  
  - `models/` – ML models with a unified interface  
  - `output/` – ML models with a unified interface  
  - `parameter_tunings/` – Classes for hyperparameter tuning  
  - `splitters/` – Classes for splitting datasets  
  - `tests/`  
    - `e2e/`Example scripts demonstrating module usage  
    - `unit/` – Unit test modules using pytest  
  - `utils/` – Utility functions  
  - `validations/` – Wrapper classes for training and prediction
- `requirements.ixt`
- `README.md`

## Setup Instructions

1. Clone the repository:  
   `git clone https://github.com/bonbonsan/ML_COMPETITIONS.git`  
   `cd ML_COMPETITIONS`

2. Create and activate a virtual environment (Python 3.11):  
   `python3.11 -m venv venv`  
   `source venv/bin/activate`  ← macOS/Linux  
   `venv\Scripts\activate`     ← Windows

3. Install required packages:  
   `pip install -r requirements.txt`

## Running Tests

`pytest my_library/tests/unit/`

## Running in Docker (GPU-accelerated training via Paperspace)

This project supports GPU-accelerated training inside Docker, with Paperspace CORE as the recommended environment.

### ❗ Prerequisite

Before starting, set your PAPERSPACE_PUBLIC_IP in a dedicated environment file:  
 `echo 'PAPERSPACE_PUBLIC_IP=184.105.4.230' > .paperspace.env`  
 This allows IP sharing across scripts.

## Shell Script Automation (GPU Setup & Training)

The following helper scripts are provided for GPU workflows using Paperspace:

### 🔧 1. gpu_setup_and_build.sh

Run once on a new Paperspace machine to install drivers and build Docker:  
`./gpu_setup_and_build.sh`

### 🔁 2. gpu_pull_and_rebuild.sh

Use when you’ve updated Dockerfile or requirements:  
`./gpu_pull_and_rebuild.sh`

### 🚀 3. docker-run_with_port.sh

Start container and run latest code with GPU:  
`./docker-run_with_port.sh`

### 🌐 4. start_paperspace.sh

On your Mac, launch SSH tunnel for Jupyter access:  
`./start_paperspace.sh`  
Make sure `PAPERSPACE_PUBLIC_IP` is set inside the script or in `.paperspace.env`.

### 📓 5. run_jupyter-lab.sh

Run inside the Docker container to launch JupyterLab:  
`./run_jupyter-lab.sh`  
Access it at: [http://localhost:8888/lab](http://localhost:8888/lab)

## Data Sync Scripts

Use the following scripts to transfer training data and results:

### ⬆️ 6. push_data_to_container.sh

From Mac → Paperspace:  
`./push_data_to_container.sh`

### ⬇️ 7. pull_data_from_container.sh

From Paperspace → Mac:  
`./pull_data_from_container.sh`

## Environment Configuration (.env)

To control GPU/CPU usage and logging, edit `.env` (copied from `.env.example` ):

`PYTHONPATH=.`  
`NVIDIA_VISIBLE_DEVICES=all`  
`USE_GPU=True`  
`LOG_LEVEL=INFO`  
`SAVE_LOG_TO_FILE=False`  
`ENABLE_TIMEIT=True`

  ⚠️ `.env` and `.paperspace.env` are excluded from Git tracking.

## Notes

- `documents/` contains external learning resources such as cheat sheets and slides.
- `my_library/` contains reusable, modular ML components.
- Unit tests are under `my_library/tests/unit/`, and example pipelines are under `e2e/`.

## License

This repository is published for personal and educational use.  
