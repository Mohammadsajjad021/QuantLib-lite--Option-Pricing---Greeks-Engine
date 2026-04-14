from setuptools import setup, find_packages

setup(
    name="quant-lite-option-pricing",
    version="0.1.0",
    author="Moe Moradi",
    author_email="moradimohammadsajjad@gmail.com",
    description="A Python library for option pricing, Greeks, and Monte Carlo simulation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mohammadsajjad021/QuantLib-lite--Option-Pricing---Greeks-Engine.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "matplotlib>=3.7"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=24.0",
            "flake8>=6.0",
            "jupyter>=1.0"
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)