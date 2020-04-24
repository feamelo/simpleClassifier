# PS-Einstein

This is a small test to check the basic understand of machine learning developed by HIAE.

I recommend the local compilation of the jupyter files since some graphs were plotted using the *Plotly* module which is not rendered in gitlab preview.

## Installation

Install python
```
sudo apt-get update
sudo apt-get install python3.6
```

Create a virtual environment called "venv" and activate it
```
python3 -m venv venv
source venv/bin/activate
```

Install the required packages
```
python3 -m pip install -r requirements.txt
```

## Usage 

Run the exploratory analysis notebook
```
jupyter notebook exploratory_analysis.ipynb
```

Run the classification notebook
```
jupyter notebook classification.ipynb
```

## Overview

The project is divided into two notebooks:

### Exploratory analysis

This type of analysis aims to provide an understanding of the data, using numerical and graphical methods, providing an in-depth view for the application of the models and consequently the extraction of knowledge.

![3dplot](images/3dplot.gif?raw=true)

### Classification

Four models were implemented for solving the *dfPoints* classification problem:
1. Logistic Regression (proposed)
2. Decision Tree
3. Random Forest
4. Support Vector Machines

The metrics are compared for selecting the one with the best accuracy.
![3dplot](images/metrics.png?raw=true)

The *Area Under the Curve* graph was used to assess the quality of the models, evaluating both the performance and the overfitting, providing information to choose the Random Forest as the best model among those evaluated.
![3dplot](images/auc.png?raw=true)