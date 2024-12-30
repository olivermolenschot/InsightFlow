# InsightFlow

InsightFlow is a comprehensive machine learning solution tailored to finance time series data. It walks you through the full lifecycle of an ML project:

    Data Ingestion & Preprocessing
        Accepts CSV or API inputs with OHL(C)V columns (and optionally a label column).
        Cleans, merges, and formats the data based on user-defined parameters.

    Feature Engineering & Model Training
        Creates lagged features, rolling averages, or technical indicators.
        Trains a classification or regression model (e.g., scikit-learn, PyTorch) that outputs a simple “buy” or “not buy” signal.
        Saves the trained model artifact for reproducibility.

    Serving Inference
        Offers a REST API endpoint (e.g., FastAPI) for real-time predictions.
        Also includes CLI commands for batch or daily predictions.
        Easily integrate with scheduling tools (Airflow, cron) if you want to automate.

    Deployment & (Optional) MLOps
        Containerize with Docker for a production-like setup.
        Hook into CI/CD (GitHub Actions) for automated testing and deployment.
        Track experiments and model versions (MLflow, Weights & Biases).

Key Features

    Flexible Schema: As long as your CSV or DataFrame has Open, High, Low, Close, and Volume (plus an optional Label column), InsightFlow can ingest it.
    Parametrized Commands: Choose the model type (random forest, logistic regression, or PyTorch-based) and pass hyperparameters via CLI arguments or config.
    Full Transparency: All stages—data, code, and results—are in your control, making it a great learning resource for building production-grade ML.
    Extendable: Adjust or inherit from PyTorch classes if you want custom neural network architectures or specialized training logic.
