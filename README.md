# group_99

The primary goal of this project is to develop and optimize a deep learning-based classifier for the Sea Animals dataset ([Sea Animals dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)). This dataset consists of images of different sea animal species, and we aim to classify them accurately. The dataset provides a rich source for multi-class classification tasks, with 23 different classes of sea animals to train for classification.

To achieve this, we will leverage PyTorch Lightning and take inspiration from the model found in the Sea Animals PyTorch Lightning CNN implementation ([Sea Animals PyTorch Lightning CNN](https://www.kaggle.com/code/stpeteishii/sea-animals-pytorch-lightning-cnn)). This framework provides a structured approach to deep learning, simplifying the training process while improving model performance through well-structured components like data loaders, optimizers, and validation metrics.

Our starting point will be the CNN architecture designed for the Sea Animals dataset in the PyTorch Lightning framework, utilizing TIMM as our third-party framework with ResNet18. This model has shown solid results in classifying images, and we will enhance its capabilities.

The dataset will be used with the provided training and validation splits, where preprocessing will include resizing and normalization with torch vision transform. This will help enhance the model’s ability to generalize to unseen examples.

In summary, our project will leverage PyTorch Lightning, TIMM, and modern deep learning techniques to develop a high-performing classifier for the Sea Animals dataset using CNNs like ResNet18.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── tests.yaml
│       ├── codecheck.yaml
│       ├── pre_commit.yaml
│       └── docker_build.yaml
├── configs/                  # Configuration files
├── datadrift/                # Data drift test
│   └── data_drift.py
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   ├── Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── frontend.py
│   │   │   └── main.py
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── command.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   └── train.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
