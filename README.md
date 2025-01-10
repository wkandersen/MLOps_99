# group_99
The primary goal of this project is to develop and optimize a deep learning-based classifier for the Food-101 dataset (https://www.kaggle.com/datasets/kmader/food41/data). This dataset consists of 101 distinct classes of dishes, with 1,000 images per class, making it a robust benchmark for multi-class classification tasks in computer vision. The classifier we aim to build will effectively distinguish between these 101 classes, achieving high accuracy and robust performance across diverse data samples.

To achieve this, we will leverage PyTorch with "PyTorch-ResNet50-84%" (https://www.kaggle.com/code/pranshu15/pytorch-resnet50-84) as our framework. Additionally, we plan to incorporate timm (Torchvision Image Models) as our third-party PyTorch-based package to streamline access to pre-trained models and enhance our model's performance through efficient feature extraction and transfer learning techniques.

Our starting point will be the ResNet50-based model implementation. This model has already demonstrated promising results on the Food-101 dataset, achieving an 84% classification accuracy. Building upon this foundation, we aim to fine-tune the ResNet50 architecture using advanced techniques like data augmentation, learning rate scheduling, and regularization.

To further expand our exploration, we plan to experiment with convolutional neural networks (CNNs) beyond ResNet50. This includes investigating novel architectures such as EfficientNet, DenseNet, or even custom CNNs designed to better capture the unique features of the Food-101 dataset. These experiments will help us identify the optimal architecture for our classification task.

For the initial dataset, we will use the provided training and validation splits of the Food-101 dataset. The images will undergo preprocessing, including resizing, normalization, and augmentation, to enhance the model's ability to generalize. While the focus will be on the dataset as-is, we may explore additional data-cleaning techniques.

To evaluate the model we will look at the loss and accuracy of the model. We may also compare the model with a baseline model, ResNet50, using McNemar's test. 
In summary, our project combines PyTorch’s robust framework, the flexibility of timm, and modern deep learning techniques to create a high-performing classifier for the Food-101 dataset.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
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
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
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
