from setuptools import setup, find_packages

setup(
    name="dual_spls",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dual Sparse Partial Least Squares in Python",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn"  # for convenience in cross-validation or splitting
    ],
    python_requires=">=3.7",
)
