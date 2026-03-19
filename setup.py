from setuptools import setup, find_packages

setup(
    name="cs-273p-final-project",
    version="0.1.0",
    description="Multi-Task Learning for Predictive Maintenance (CS273P Final Project)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
    ],
)
