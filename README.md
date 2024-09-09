# SASRec model
This repository contains adaptations to the ["Self-Attentive Sequential Recommendation" (Kang et al. 2018]( https://doi.org/10.48550/arXiv.1808.09781)
(SASRec)model  for personalized recommendations on the ZDF-Mediathek (more information [here](https://algorithmen.zdf.de/awf/dkdi/model-card) and [here](https://algorithmen.zdf.de/awf/dkdi)), by modifying loss function and sampling strategies:

* [TRON negative sampling and loss function (Wilm et.al  2023)](https://doi.org/10.1145/3604915.3610236)
* [gbce loss function (Petrov et al. 2023)](https://doi.org/10.48550/arXiv.2308.07192)
* [Popularity based negative sampling (Hidasi et al. 2015)](https://doi.org/10.48550/arXiv.1511.06939)

Additionally it contains:
* ZDF specific data preprocessing
* a SageMaker pipeline for training, inference and hyperparameter optimization

***

## Environment Setup
1. Install python 3.8 and make sure it is used in your current shell. You can use pyenv for this: https://github.com/pyenv/pyenv
2. Install poetry: https://python-poetry.org/docs/#installation (build a wheel )
3. activate poetry shell
   ``` 
   poetry shell
   ```
3. Install pa-base from https://github.com/zdf-opensource/recommendations-pa-base. See instructions in the README (build a wheel and install it with pip with the poetry shell activated)
4. Install remaining dependencies using pip in a poetry shell. Installing directly using poetry does not work for scikit-surprise, lightfm and recommenders due to pep517 incompatibility. Some of the dependencies won't work on a arm64 architecture.
    ```
    poetry export -f requirements.txt --without-hashes --with-credentials | pip install -r /dev/stdin
    ```

***

## Maintainers
This project is maintained by the Personalization & Automation Team at ZDF.

***

## License
See [LICENSE](LICENSE).

