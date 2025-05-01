This repository contains the code and models for our CS182 project report.
This repository is based on the repo: https://github.com/dtsip/in-context-learning.

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://github.com/Wanderer09/CS182-project/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/Wanderer09/CS182-project/releases/download/v1.0/models.zip
    unzip models.zip
    ```

3. If you plan to train, populate `conf/wandb.yaml` with you wandb info.

- The `SineEval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/sine2exp.yaml` for a quick training run.
- Please notice that if you want to train, use sine2exp.yaml. Do not use sine.yaml. When evaluating, please change the "mode" choice in Sine2Exp.evaluate in tasks.py to "sine"!
- If you met ImportError undefined symbol: iJIT_NotifyEvent, please check this issue: https://github.com/pytorch/pytorch/issues/123097.

