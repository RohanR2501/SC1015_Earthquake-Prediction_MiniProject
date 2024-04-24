# SC1015 MiniProject: Earthquake Magnitude Prediction

### Important!: The Data Preprocessing and Exploration ipynb file isn't being completely displayed on github due to some issues with importing the folium library. Please donwload the databases(earthquake database and tectonic plates) and the raw code to view/run it on your local machine

## Introduction/Practical Motivation:

Major earthquakes pose significant risks to lives and property, emphasizing the need for accurate prediction models. This project focuses on predicting earthquake magnitudes(exact and range) using key parameters like geographical coordinates, depth, and proximity to tectonic plate boundaries.

## Objective/Problem statement:

To determine the most effective time forecasting/ regression model for predicting major earthquakes’ magnitude and the most effective classification model to predict the RANGE of  major earthquakes’ magnitudes based on geographical coordinates (latitude/longitude) and depth.
 
## About Code: 

The entire codebase has been split into three different parts.
Please follow the flow of project as provided below:

1. [Data Preprocessing and Exploratory Analysis](https://github.com/RohanR2501/SC1015_DataScienceProject_Earthquake-Prediction/blob/main/Data%20Preprocessing%20and%20Exploratory%20Analysis.ipynb)
2. [Time series forecasting and regression based models](https://github.com/RohanR2501/SC1015_DataScienceProject_Earthquake-Prediction/blob/main/Time%20forecasting%20and%20Regression%20Models.ipynb)
3. [Classification based models and conclusion](https://github.com/RohanR2501/SC1015_DataScienceProject_Earthquake-Prediction/blob/main/Classification%20Models%20And%20Conclusions.ipynb)

(Comments and Statistical Visualization included wherever necessary)

## Datasets used 

- Earthquake Dataset (magnitude 5.5 and higher from 1965 to 2016)
- Tectonic Plate Boundaries Data
 
## Libraries Used

- Pandas
- NumPy
- Folium
- Statsmodels
- Time
- Scikit-learn (sklearn)
- Matplotlib
- TensorFlow
- Seaborn
- Scikeras (wrapper for Keras models)

## How to Run the Code

To utilize the code provided in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Ensure that the required libraries are installed, as listed above.
3. Execute the provided Python script or Jupyter Notebook containing the data cleaning and preprocessing code.

## Models Used

### Time Series Forecasting and Regression Based Models

1. ARIMA (AutoRegressive Integrated Moving Average)
2. Gradient Boosting Regression (GBR)
3. Support Vector Regression (SVR)
4. Random Forest Regressor (RFR)
5. Custom Neural Network (NN)

### Classification Models

1. Custom Neural Network (NN)
2. Gradient Boosting Classifier (GBC)

## Results

### Time series forecasting and regression based models’ results
| Model | MSE (Mean Squared Error) | MAE (Mean Absolute Error) | Time Taken (seconds) |
|-------|---------------------------|----------------------------|----------------------|
| ARIMA | 0.204 | 0.378 | 0.223 |
| SVR | 0.192 | 0.293 | 1.869 |
| RFR | 0.195 | 0.322 | 0.115 |
| GBR | 0.174 | 0.307 | 0.005 |
| NN | 0.177 | 0.314 | 9.183 |

### Classification Results
| Model | Accuracy | F1 Score | Time Taken (seconds) |
|-------|----------|----------|----------------------|
| NN | 0.750 | 0.656 | 0.020
| GBC | 0.751 | 0.652 | 0.019 |


## Conclusions

- Time series forecasting and regression based models:
  - Gradient Boosting Regressor (GBR) emerges as the most effective regression model, achieving the lowest MSE and MAE.
  - GBR demonstrates strong performance in minimizing prediction errors, making it suitable for directly predicting earthquake magnitude.
  
- Classification Models:
  - Gradient Boosting Classifier (GBC) is identified as the most effective classification model, exhibiting good accuracy and efficient computational time, although just slightly better than Custom Neural Networks (NN)
  - GBC is recommended for classifying range of magnitude of significant earthquake occurrences based on geographical coordinates and depth.

## References

- Earthquake dataset (https://www.kaggle.com/datasets/usgs/earthquake-database/data)
- Tectonic Plate dataset(https://www.kaggle.com/datasets/cwthompson/tectonic-plate-boundaries)
- Tensorflow (https://www.tensorflow.org/)


