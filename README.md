# group_99

The primary goal of this project is to develop and optimize a deep learning-based classifier for the Sea Animals dataset ([Sea Animals dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)). This dataset consists of images of different sea animal species, and we aim to classify them accurately. The dataset provides a rich source for multi-class classification tasks, with 23 different classes of sea animals to train for classification.

To achieve this, we will leverage PyTorch Lightning and take inspiration from the model found in the Sea Animals PyTorch Lightning CNN implementation ([Sea Animals PyTorch Lightning CNN](https://www.kaggle.com/code/stpeteishii/sea-animals-pytorch-lightning-cnn)). This framework provides a structured approach to deep learning, simplifying the training process while improving model performance through well-structured components like data loaders, optimizers, and validation metrics.

Our starting point will be the CNN architecture designed for the Sea Animals dataset in the PyTorch Lightning framework, utilizing TIMM as our third-party framework with ResNet18. This model has shown solid results in classifying images, and we will enhance its capabilities.

The dataset will be used with the provided training and validation splits, where preprocessing will include resizing and normalization with torch vision transform. This will help enhance the model’s ability to generalize to unseen examples.

In summary, our project will leverage PyTorch Lightning, TIMM, and modern deep learning techniques to develop a high-performing classifier for the Sea Animals dataset using CNNs like ResNet18.

## Project structure

The directory structure of the project looks like this:
```txt
└── data.dvc
├── datadrift
    └── data_drift.py
├── dockerfiles
    └── api.dockerfile
    └── Dockerfile.dockerfile
    └── train.dockerfile
├── docs
    └── mkdocs.yaml
    └── README.md
    ├── source
        └── index.md
└── environment.yaml
├── google-cloud-sdk [Excluded]
└── labels.csv
└── LICENSE
├── lightning_logs [Excluded]
├── models
    └── best-model-epoch=04-val_loss=0.77.ckpt
    └── best-model-epoch=04-val_loss=2.27.ckpt
    └── best-model-epoch=13-val_loss=2.13.ckpt
    └── best-model-epoch=18-val_loss=2.04.ckpt
    └── best-model-epoch=19-val_loss=2.06.ckpt
    └── bestmodel.ckpt
├── notebooks
    ├── lightning_logs [Excluded]
    └── test.ipynb
├── outputs
    ├── 2025-01-12
        ├── 11-52-10
        ├── 14-22-01
        ├── 14-22-30
        ├── 14-27-34
        ├── 14-29-18
        ├── 14-31-44
    ├── 2025-01-15
        ├── 11-05-56
        ├── 11-11-51
        ├── 11-13-10
        ├── 11-33-26
        ├── 11-40-53
            └── train.log
        ├── 15-33-28
        ├── 15-33-51
        ├── 15-41-40
        ├── 17-10-10
        ├── 17-10-44
        ├── 17-14-33
    ├── 2025-01-19
        ├── 10-21-29
        ├── 10-21-40
        ├── 10-23-04
        ├── 10-23-57
        ├── 11-34-57
        ├── 11-40-22
        ├── 11-40-43
        ├── 11-44-22
        ├── 11-47-49
        ├── 11-49-50
        ├── 13-42-05
        ├── 13-42-16
        ├── 13-43-38
        ├── 13-43-58
        ├── 13-46-52
        ├── 13-54-48
        ├── 13-56-30
        ├── 13-57-32
        ├── 14-00-33
        ├── 14-00-49
        ├── 14-01-33
        ├── 14-02-37
        ├── 15-46-26
        ├── 15-47-32
        ├── 15-47-43
        ├── 15-48-10
        ├── 15-53-25
        ├── 15-54-25
        ├── 15-54-57
        ├── 15-55-09
        ├── 15-55-55
        ├── 16-04-56
        ├── 16-05-15
        ├── 16-06-35
        ├── 16-08-00
        ├── 16-08-10
        ├── 16-08-38
        ├── 16-09-14
        ├── 16-09-50
        ├── 16-13-46
        ├── 16-14-10
        ├── 16-22-04
        ├── 16-22-18
        ├── 16-22-55
        ├── 16-30-37
        ├── 16-30-51
        ├── 16-31-28
    ├── 2025-01-20
        ├── 09-49-28
        ├── 09-52-02
        ├── 09-54-30
        ├── 09-54-50
        ├── 09-56-34
        ├── 09-57-35
        ├── 10-00-52
        ├── 10-02-42
        ├── 10-08-31
        ├── 10-14-37
        ├── 10-19-08
        ├── 10-21-51
        ├── 10-26-10
        ├── 10-26-50
        ├── 10-27-58
        ├── 10-28-26
        ├── 10-29-18
        ├── 10-34-32
        ├── 10-35-21
        ├── 10-35-54
        ├── 10-36-58
        ├── 16-53-30
        ├── 16-53-44
        ├── 16-54-42
    ├── 2025-01-22
        ├── 10-17-18
        ├── 10-17-54
        ├── 10-23-17
        ├── 10-36-21
        ├── 10-36-43
        ├── 10-37-33
        ├── 10-47-58
        ├── 10-48-49
        ├── 12-58-23
        ├── 12-59-45
        ├── 13-02-45
        ├── 13-03-03
        ├── 13-03-22
        ├── 13-04-48
        ├── 13-08-33
        ├── 13-45-57
        ├── 22-14-26
        ├── 22-40-43
    ├── 2025-01-23
        ├── 09-28-09
            └── train.log
        ├── 09-34-15
        ├── 09-35-21
        ├── 09-35-42
        ├── 09-36-08
        ├── 09-36-20
        ├── 09-37-29
        ├── 09-39-16
        ├── 09-40-44
        ├── 09-41-36
        ├── 09-42-01
        ├── 16-28-26
        ├── 16-30-40
        ├── 16-32-34
        ├── 16-52-58
        ├── 16-56-17
        ├── 17-00-06
        ├── 17-09-01
        ├── 18-01-55
    ├── 2025-01-24
        ├── 09-07-39
└── pyproject.toml
└── README.md
└── report.html
├── reports
    └── dir.ipynb
    ├── figures
        └── artifact_Q20.png
        └── bucket_Q19.png
        └── classification_report.txt
        └── confusion_matrix.png
        └── mlops_overview.drawio.png
        └── wandb_logged_results.png
    └── README.md
    └── report.py
└── requirements.txt
└── requirements_dev.txt
├── runs
    ├── Jan12_14-22-30_william-zenbookum5401qab
        └── events.out.tfevents.1736688150.william-zenbookum5401qab.27726.0
        └── events.out.tfevents.1736688150.william-zenbookum5401qab.27726.1
    ├── Jan12_14-31-44_william-zenbookum5401qab
        └── events.out.tfevents.1736688704.william-zenbookum5401qab.31326.0
        └── events.out.tfevents.1736688704.william-zenbookum5401qab.31326.1
└── setup.py
├── src
    ├── group_99
        ├── api
            └── frontend.py
            └── main.py
            └── __init__.py
        └── command.py
        ├── config
            └── config.yaml
        └── data.py
        └── evaluate.py
        └── model.py
        ├── outputs
            ├── 2025-01-20
                ├── 16-52-02
        ├── runs
            ├── Jan12_14-20-31_william-zenbookum5401qab
                └── events.out.tfevents.1736688031.william-zenbookum5401qab.26987.0
            ├── Jan12_14-21-25_william-zenbookum5401qab
                └── events.out.tfevents.1736688085.william-zenbookum5401qab.27159.0
                └── events.out.tfevents.1736688085.william-zenbookum5401qab.27159.1
            ├── Jan12_14-21-37_william-zenbookum5401qab
                └── events.out.tfevents.1736688097.william-zenbookum5401qab.27244.0
                └── events.out.tfevents.1736688097.william-zenbookum5401qab.27244.1
        └── train.py
        ├── wandb
            ├── latest-run
                └── run-gwsiqfcc.wandb
            ├── run-20250120_165202-gwsiqfcc
                └── run-gwsiqfcc.wandb
            ├── sweep-n4h3hv4s
                └── config-4gh4oipm.yaml
                └── config-ry43z1mu.yaml
                └── config-y4hek72r.yaml
        └── __init__.py
        ├── __pycache__
            └── api.cpython-311.pyc
            └── data.cpython-311.pyc
            └── model.cpython-311.pyc
            └── __init__.cpython-311.pyc
    └── __init__.py
    ├── __pycache__
        └── __init__.cpython-311.pyc
└── src.tar.gz
└── tasks.py
├── tests
    └── test_data.py
    └── test_model.py
    └── __init__.py
    ├── __pycache__
        └── test_api.cpython-311-pytest-8.3.4.pyc
        └── test_data.cpython-311-pytest-8.3.4.pyc
        └── test_model.cpython-311-pytest-8.3.4.pyc
        └── __init__.cpython-311.pyc
└── trainonline.py
├── vertex_package
    ├── config
        └── config.yaml
    └── data.py
    └── model.py
    └── requirements.txt
    └── train.py
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
