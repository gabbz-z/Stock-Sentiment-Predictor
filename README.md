# Stock-Sentiment-Predictor

StockSentimentPredictor is a machine learning project designed to predict stock market movements using financial news sentiment analysis and historical stock data. By integrating Natural Language Processing (NLP) and market trend analysis, the project provides insights into stock price trends, making it a practical tool for trading strategy optimization.

## Features
* Sentiment Analysis: Scores financial news headlines and descriptions.
* Stock Data Integration: Combines historical stock data and technical indicators.
* Predictive Modeling: Uses machine learning algorithms (e.g., Random Forest) to classify price movements.
* Class Balancing: Handles data imbalance using SMOTE.
* Visual Insights: Provides detailed visualizations, including confusion matrices and feature importance plots.
* Cross-Validation: Ensures the model’s robustness with k-fold cross-validation.

## How It Works
1. Data Collection:
    * Financial news headlines and descriptions are sourced from APIs or CSV files.
    * Historical stock data is merged with sentiment scores.
2. Data Preprocessing:
    * Performs sentiment analysis on news.
    * Generates technical indicators like moving averages, volatility, and lagged prices.
3. Model Training:
    * Trains a Random Forest Classifier on the processed dataset.
    * Labels data based on price changes (up or down).
4. Evaluation:
    * Evaluates performance using metrics like accuracy, precision, recall, and F1-score.
    * Validates the model through cross-validation.
5. Visualization:
    * Confusion matrix to show prediction performance.
    * Feature importance to highlight key predictive factors.
    * Actual vs. Predicted labels to compare model output.

  ## Results
This section provides an example of how the program was run on Alphabet Inc. (GOOGL) stock data. The following outcomes were achieved:
* Accuracy: Achieved 82% accuracy on test data.
* Insights:
    * Volatility and daily sentiment were the most influential predictors.
    * Model demonstrated robust performance with balanced precision and recall.
Example Outputs:
1. Confusion Matrix:
    * Visualized to show correct and incorrect predictions.
2. Feature Importance:
    * Highlighting key features driving the predictions.
3. Actual vs Predicted Trends:
    * A plot comparing the model's predictions with the actual stock movements.


