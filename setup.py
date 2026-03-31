"""
Setup script for Boat Random Walk Simulator
For use with Claude Code and professional development
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="boat-random-walk-simulator",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="A simulation platform for autonomous robotic boat sampling using Correlated Random Walk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/boat-simulator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "boat-simulator=boat_simulator.cli:main",
            "run-simulation=boat_simulator.examples.run_basic:main",
        ],
    },
    include_package_data=True,
    package_data={
        "boat_simulator": ["config/*.yaml", "data/*.csv"],
    },
)
