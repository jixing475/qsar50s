"""
Setup configuration for QSAR50S package.
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qsar50s",
    version="1.0.0",
    author="Jixing Liu",
    author_email="your.email@example.com",
    description="Machine Learning-Based Virtual Screening for 50S Ribosomal Inhibitors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qsar50s",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qsar50s-preprocess=qsar50s.data.preprocessing:main",
            "qsar50s-fingerprints=qsar50s.features.fingerprints:main",
            "qsar50s-train=qsar50s.models.qsar_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qsar50s": [
            "config/*.yaml",
            "data/external/PaDEL-Descriptor/*",
        ],
    },
    zip_safe=False,
    keywords="qsar, machine-learning, virtual-screening, cheminformatics, drug-discovery",
    project_urls={
        "Bug Reports": "https://github.com/your-username/qsar50s/issues",
        "Source": "https://github.com/your-username/qsar50s",
        "Documentation": "https://qsar50s.readthedocs.io/",
    },
)