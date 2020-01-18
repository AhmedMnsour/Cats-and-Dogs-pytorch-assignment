# Cat vs Dog Classification With Pytorch Assignment
This repo is an assignment as part of the African Master's in Machine Intelligence [AMMI](https://aimsammi.org/) The objective is to use a Convolutional Neural Network model to distinguish dogs from cats.

## Problem description
The problem is to write an algorithm to classify images into two categories: dog vs cat.  This is easy for humans. We will in this project make our computers able to do this kind of classification. 

### Installation

- Clone this repo:
```bash
git clone https://github.com/AhmedMnsour/Cats-and-Dogs-pytorch-assignment
cd Cats-and-Dogs-pytorch-assignment
pip install requirements.txt
```
# The Dataset

- Download our the data from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
- Extract the train.zip file then rename to dataset.

```bash
    cd dataset
    
    mkdir -p train/dog
    mkdir -p train/cat
    mkdir -p test/dog
    mkdir -p test/cat
    
    #Randomly split into train and test, 
    
    find . -name "cat*" -type f | shuf -n10000 | xargs -I file mv file train/cat/
    find . -maxdepth 1 -type f -name 'cat*'| xargs -I file mv file test/cat/
    
    find . -name "dog*" -type f | shuf -n10000 | xargs -I file mv file train/dog/
    find . -maxdepth 1 -type f -name 'dog*'| xargs -I file mv file test/dog
    
    
```


### Run the main file
```python
python main.py
```
