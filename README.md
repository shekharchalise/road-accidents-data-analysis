# Road Accidents in US Data Analysis

This project is a basic data analysis and prediction for our Master's Big Data Technology Course

## Requirements
```
- Python 3.7 or higher
- Jupyter Server
- Numpy
- Pandas
- Sklearn
- Keras
- Matplotlib
- Geopandas
- Seaborn
- Tabulate
- Descartes
- Statsmodels
```
## Installation
- We would suggest using `anaconda` to install these packages because some packages like geopandas has lots of dependencies
- We used `Jupyter notebook` to do our code for analysis, so, we would suggest to use the similar packages so that you can see what is happening in each step

## Dataset
- Download Dataset [here](https://www.kaggle.com/sobhanmoosavi/us-accidents/download/p44lgYpA1uUpTDM3fEsI%2Fversions%2F7rBcGxF5y4DRaC7WGHpu%2Ffiles%2FUS_Accidents_May19.csv?datasetVersionNumber=1)
- Rename the dataset file to `US_Accidents_May19.csv`

## Data Pre-Processsing
- Make sure you are in the same directory where the dataset file resides
-   ```
    python data-preprocessing.py
    ```
- This will create a `preprocessed-file.csv`

## Predictions

### With Weather Condition Features
- You can either run the Jupyter Notebook or just the python script
- The notebook for running predictions **with weather condition features is `prediction.ipynb`**
- The python script for running predictions **with weather condition features is `predicition.py`**
- To run the script
    ```
    python predicitions.py
    ```
- Result will be saved `results` directory
- Inside the directory you will be able to find classification report and confusion matrix images for each classifier and overall score comparision CSV file `prediction-score.csv`.

### Without Weather Condition Features
- You can either run the Jupyter Notebook or just the python script
- The notebook for running predictions **without weather condition features is `predictions_without_weather.ipynb`**
- The python script for running predictions **without weather condition features is `predictions_without_weather.py`**
- To run the script
    ```
    python predictions_without_weather.py
    ```
- Result will be saved `results-without-weather` directory
- Inside the directory you will be able to find classification report and confusion matrix images for each classifier and overall score comparision CSV file `prediction-score.csv` as before.