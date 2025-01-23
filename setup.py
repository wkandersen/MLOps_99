from setuptools import find_packages, setup

setup(
    name="group_99",
    version="0.1",
    packages=find_packages(include=["src*", "src.*"]),
    install_requires=[
        "google-cloud-storage",  # Example dependency, add your own from requirements.txt
        # Add more dependencies here if necessary
    ],
    entry_points={
        "console_scripts": [
            "train=src.group_99.train:main",  # Entry point for the training script
        ],
    },
)
