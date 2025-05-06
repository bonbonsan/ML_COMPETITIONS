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
  - `putput/` – ML models with a unified interface  
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

## Running in Docker (GPU-accelerated training)

You can also run the entire library inside a Docker container with GPU support (e.g., on Paperspace or AWS).

1. Create a .env file from the template  
   `cp .env.example .env`  
   Edit the .env file to configure runtime behavior:  
   `USE_GPU=False     # Set to True when using GPU`  
   `PYTHONPATH=.      # Required for resolving my_library imports`  
   ⚠️ `.env` is excluded from Git and Docker build (.gitignore, .dockerignore) and should not be committed.

2. Build the Docker image  
   `docker build -t ml-gpu .`

3. Run the container (with GPU support if available)  
   `docker run --env-file .env -p 8888:8888 -v $PWD:/workspace -it ml-gpu`  
   If using GPU (e.g., on Paperspace), you may also add: `--gpus all` to enable GPU access. For example:  
   `docker run --gpus all --env-file .env -p 8888:8888 -v $PWD:/workspace -it ml-gpu`
  
   - --env-file .env: Injects environment variables into the container  
   - -v $PWD:/workspace: Mounts the project into /workspace inside the container  
   - -it: Runs interactively with terminal access  

4. Inside the container, You can run training or testing as usual:  
   `pytest my_library/tests/unit/`  
   `python projects/train.py`  
   To check environment variables:  
   `echo $USE_GPU`  
   `echo $PYTHONPATH`  

5. To run Jupyter Notebook:  
   Inside the container, a helper script is pre-installed to simplify starting Jupyter Lab.  
   Simply run: `jupyter-lab.sh`

6. Exiting the container  
   Simply type: `exit`

## Notes

- `documents/` contains external learning resources such as cheat sheets and slides.
- The modules under `my_library/` are organized to cover all stages of a machine learning workflow.
- Sample codes are placed in `my_library/tests/e2e/`, while unit tests are in `my_library/tests/unit/`.

## License

This repository is published for personal and educational use.  
