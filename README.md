<p align="center"><img width="17%" src="docs/RecLib.png" /></p>

[![Build Status](https://dev.azure.com/Torch-AI/RecLib/_apis/build/status/TorchAI.reclib?branchName=master)](https://dev.azure.com/Torch-AI/RecLib/_build/latest?definitionId=1&branchName=master)
<a style="margin: 0 5px" href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/github/license/tingkai-zhang/reclib" alt="GitHub license"></a>

UPDATE: This project is not under mantainance

a Python library for recommender system, which is hyper-modular, extensively tested and easy to extend.
Reclib makes it easy to design and evaluate deep learning models for recommender system, along with the infrastructure to easily run them in the cloud or on your laptop.


The CLI part is under construction for now.


## Package Overview
| **reclib** | an Python library for recommender system |
| --- | --- |
| **reclib.commands** | functionality for a CLI and web service |
| **reclib.data** | a data processing module for loading and encoding datasets for representation |
| **reclib.models** | a collection of state-of-the-art models |
| **reclib.modules** | a collection of PyTorch modules for use with recommender system |
| **reclib.nn** | tensor utility functions, such as initializers and activation functions |
| **reclib.service** | a web server to that can serve demos for your models |
| **reclib.training** | functionality for training models |


## Installation

reclib requires Python 3.7.1 or later. The preferred way to install reclib is via `pip`.  Just run `pip install reclib` in your Python environment and you're good to go!

If you need pointers on setting up an appropriate Python environment or would like to install reclib using a different method, see below.

Windows is currently not officially supported, although we try to fix issues when they are easily addressed.


### Installing from source
```
git clone https://github.com/TorchAI/reclib.git
pip install reclib/
```

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for reclib.  If you already have a Python 3.7 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.7

    ```bash
    conda create -n reclib python=3.7
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use reclib.

    ```bash
    source activate reclib
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

   ```bash
   pip install reclib
   ```

That's it! You're now ready to build and train reclib models.
reclib installs a script when you install the python package, meaning you can run reclib commands just by typing `reclib` into a terminal.

You can now test your installation with `reclib test-install`.

_`pip` currently installs Pytorch for CUDA 9 only (or no GPU). If you require an older version,
please visit https://pytorch.org/ and install the relevant pytorch binary._

## Features

- Pythonic
- Easy to use
- State-of-the-art


## Models

Please refer to the documents

### MovieLens 1M

| Model | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Auc | Explained Variance | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |--- | 
| AutomaticFeatureInteraction |   |   |   |   |  |    |   |  0.7872 |   | 
| AttentionalFactorizationMachine |   |   |   |   |  |    |   |  0.77824 |   | 
| DeepCrossNetwork |   |   |   |   |  |    |   |  0.7928 |   | 
| DeepFactorizationMachine |   |   |   |   |  |    |   |  0.7928 |   | 
| FieldAwareFactorizationMachine |   |   |   |   |  |    |   |  0.7928 |   | 
| FactorizationMachine |   |   |   |   |  |    |   |  0.7928 |   | 
| FieldAwareNeuralFactorizationMachine |   |   |   |   |  |    |   |  0.7928 |   | 
| FactorizationSupportedNeuralNetwork |   |   |   |   |  |    |   |  0.7928 |   | 
| NeuralFactorizationMachine |   |   |   |   |  |    |   |  0.7928 |   | 
| ProductNeuralNetwork |   |   |   |   |  |    |   |  0.7928 |   | 
| WideAndDeep |   |   |   |   |  |    |   |  0.7928 |   | 
| ExtremeDeepFactorizationMachine |   |   |   |   |  |    |   |  0.800158 |   | 




## Issues
Everyone is welcome to file issues with either feature requests, bug reports, or general questions. As a small team with only one person, we may ask for contributions if a prompt fix doesn't fit into our roadmap. We allow users a two week window to follow up on questions, after which we will close issues. They can be re-opened if there is further discussion.

## Contributions
If you would like to contribute a larger feature, we recommend first creating an issue with a proposed design for discussion. This will prevent you from spending significant time on an implementation which has a technical limitation someone could have pointed out early on. Small contributions can be made directly in a pull request.

Pull requests (PRs) must have one approving review and no requested changes before they are merged. 

## Licence
Apache 2.0 

