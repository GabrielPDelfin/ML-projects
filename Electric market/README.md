Electricity prize prediction using XGBoost and LSTM models.

Both models take historic time-series data from the Iberian Energy Market Operator website (https://www.omie.es/es/market-results/daily/daily-market/day-ahead-price) and predict the prizes with one day in advance.

Feature engineering techniques were performed to enhance the performance of both models.

In order to predict the prizes for each time period in a day, the models utilize a sliding window that takes all the data belonging to several time periods before and make a prediction for the newest time period. 
This new value becomes a input for the sliding window to make the next prediction while the oldest data value gets discarded and so on.
