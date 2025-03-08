# Break the Captcha Challenge

Yu-Kai "Steven" Wang

Here we explore three different approach to solve the [Kaggle Break the CAPTCHA Challenge](https://www.kaggle.com/competitions/break-the-captcha-challenge/overview).

Three models for solving this challenge:
1. CNN (Convolutional Neural Network) + Cross Entropy Loss
2. CRNN (Convolutional Recurrent Neural Network) + CTC Loss
3. ViT (Vision Transformer) + CTC Loss

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

### Evaluation

See the `<model-name>_eval.ipynb` for more details.
