from setuptools import find_packages, setup

setup(
    name="group_99",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-json-logger",  # Add other dependencies from requirements.txt here
    ],
    entry_points={
        "console_scripts": [
            "train=src.group_99.train:main",
        ],
    },
)
