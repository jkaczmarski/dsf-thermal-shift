from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dsf_analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive package for analyzing Differential Scanning Fluorimetry (DSF) data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dsf-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "plotly>=4.14.0",
        "kaleido",  # For saving plotly figures
    ],
    extras_require={
        "smiles": ["rdkit"],
        "dev": [
            "pytest>=6.0.0",
            "black",
            "flake8",
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "dsf-analyze=dsf_analysis.cli.commands:analyze_command",
        ],
    },
)
