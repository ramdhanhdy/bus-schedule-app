from setuptools import setup, find_packages

setup(
    name="bus_schedule",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pandas>=2.1.3",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.2",
        "lightgbm>=4.1.0",
        "shap>=0.43.0",
        "python-dotenv>=1.0.0",
        "joblib>=1.3.2",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "scipy>=1.11.3",
        "pydantic>=2.5.1"
    ],
    python_requires=">=3.11.6",
)