# ML_COMPETITIONS

This repository is a custom-built machine learning library developed in Python 3.11.  
It is designed for competitive data science tasks (e.g., Kaggle) and provides a modular and reusable codebase with a unified interface.

## Module Descriptions

- `documents/` – Cheat sheets and slides obtained from online sources
- `my_library/` – Core machine learning library  
  - `configs/` – Configuration classes for each algorithm using dataclasses  
  - `data/` – Sample datasets collected from public sources  
  - `ensembles/` – Classes for combining predictions (ensembling)  
  - `feature_engineerings/` – Functions for feature engineering  
  - `logs/` – Output destination for log files  
  - `models/` – ML models with a unified interface  
  - `parameter_tunings/` – Classes for hyperparameter tuning  
  - `splitters/` – Classes for splitting datasets  
  - `tests/e2e/` – Example scripts demonstrating module usage  
  - `tests/unit/` – Unit test modules using pytest  
  - `utils/` – Utility functions  
  - `validations/` – Wrapper classes for training and prediction
- `requirements.ixt`
- `README.md`

## Setup Instructions

1. Clone the repository:
`git clone [https://github.com/your-username/ML\_COMPETITIONS.git](https://github.com/your-username/ML_COMPETITIONS.git)`
`cd ML\_COMPETITIONS`

2. Create and activate a virtual environment (Python 3.11):
`python3.11 -m venv venv`
`source venv/bin/activate  # On Windows: venv\Scripts\activate`

3. Install required packages:
`pip install -r requirements.txt`

## Running Tests

`pytest my_library/tests/unit/`

## Notes

- `documents/` contains external learning resources such as cheat sheets and slides.
- The modules under `my_library/` are organized to cover all stages of a machine learning workflow.
- Sample codes are placed in `my_library/tests/e2e/`, while unit tests are in `my_library/tests/unit/`.

## License

This repository is published for personal and educational use.  
