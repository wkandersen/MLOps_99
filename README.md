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
│   └── workflows/
│       ├── codecheck.yaml
│       ├── docker-build.yaml
│       ├── pre_commit.yaml
│       └── tests.yaml
├── datadrift/
│   └── data_drift.py
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   ├── Dockerfile.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   ├── README.md
│   └── source/
│       └── index.md
├── google-cloud-sdk/
├── lightning_logs/
├── models/                   # Trained models
│   ├── best-model-epoch=04-val_loss=0.77.ckpt
│   ├── best-model-epoch=04-val_loss=2.27.ckpt
│   ├── best-model-epoch=13-val_loss=2.13.ckpt
│   ├── best-model-epoch=18-val_loss=2.04.ckpt
│   ├── best-model-epoch=19-val_loss=2.06.ckpt
│   └── bestmodel.ckpt
├── notebooks/                # Jupyter notebooks
│   ├── lightning_logs/
│   └── test.ipynb
├── outputs/
├── reports/                  # Reports
│   ├── figures/
│   ├── README.md
│   └── report.py
├── runs/
├── src/                      # Source code
│   ├── __init__.py
│   ├── group_99/
│   │   ├── api/
|   │   │   ├── __init__.py
|   │   │   ├── frontend.py
|   │   │   └── main.py
│   │   ├── config/
│   │   ├── outputs/
│   │   ├── runs/
│   │   ├── wandb/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── .gitattributes
├── .gitignore
├── pre-commit-config.yaml
├── data-to-csv.py
├── data.dvc
├── environment.yaml
├── labels.csv
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── report.html
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
├── setup.py
├── src.tar.gz
├── tasks.py                  # Project tasks
└── trainonline.py
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
