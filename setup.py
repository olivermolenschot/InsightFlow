from setuptools import setup, find_packages
import pathlib

# Get the directory containing this file
HERE = pathlib.Path(__file__).parent

# Read the README file for the long description
README = (HERE / "README.md").read_text(encoding="utf-8")
setup(
    name="InsightFlow",  
    version="0.1.0",
    author="Oliver Molenschot",  
    author_email="ojm24@cornell.com",  
    description="A comprehensive library for data ingestion, preprocessing, feature engineering, model training, and serving inference with deployment and MLOps capabilities.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/olivermolenschot/InsightFlow",  
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7, <4',
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.23.0",
        "torch>=1.13.1",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "mlops": [
            "mlflow>=1.20.0",
            "apache-airflow>=2.0.0",
        ],
        "docker": [
            "docker>=5.0.0",
        ],
    },
    include_package_data=True,  
    license="Apache 2.0",
    keywords="data-ingestion preprocessing feature-engineering model-training inference serving-deployment mlops",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/insightflow/issues",
        "Source": "https://github.com/yourusername/insightflow",
    },
)
