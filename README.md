# Break the Captcha Challenge

Yu-Kai "Steven" Wang

Here we explore three different approach to solve the [Kaggle Break the CAPTCHA Challenge](https://www.kaggle.com/competitions/break-the-captcha-challenge/overview).

Three models for solving this challenge:
1. CNN (Convolutional Neural Network) + Cross Entropy Loss
2. CRNN (Convolutional Recurrent Neural Network) + CTC Loss
3. ViT (Vision Transformer) + CTC Loss

A demo webapp build using built using `postgres`, `fastapi`, `React` and `Docker` is also available to test out the model. See section "Run Demo with Docker" for more details.

### Dependencies

All code in this repo is tested on python 3.12 only.

```
conda create --name captcha python=3.12
```

Install the dependencies required to train and run the models:

```
pip install -r requirements.txt
```

### Dataset

All dataset used in training these models are provided by the Kaggle challenge. 
This includes 30k labelled training data and 10k unlabeled testing data.

Download the data from the Kaggle challenge link and unzip it in the local "dataset" folder.

Your final project directory should look like this:

```
src/
dataset/
...
```

### Training the models

Run the training script written for the corresponding models:

```
python train_<model-name>.py
```

For example, to run the training code for the ViT model:

```
python train_vit.py
```

### Weights

You can download my model weights [here](https://drive.google.com/file/d/19E32XYX-TuEMgJQHBd3BQjauPuou7vIY/view?usp=drive_link).

Unzip the downloaded weights at project root under `checkpoints`.

### Evaluation

See the `<model-name>_eval.ipynb` for more details.

### Run Demo with Docker

See [`README.docker.md`](/README.docker.md) for more details.