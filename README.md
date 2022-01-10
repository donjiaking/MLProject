# MLProject: Human Face Expression Recognition

## Prerequisite

1. Download FER2013 dataset from [kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and put `train.csv` `test.csv` under `./dataset`. Note that you need to manually add label column in `test.csv`.
2. Then run `python generate_data.py`, which will create training, validation and test images set under `./dataset/images`.

## How to run
- **SVM**: run `python svm.py`. Refer to the code to change some settings.
- **CNN**: run `python train_cnn.py` to train. run `python test_cnn.py` to test. You can set your own arguments (please refer to the code). `demo.py` provides a simple demo to predict expression probability given an image.