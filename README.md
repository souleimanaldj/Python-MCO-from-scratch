# Python | Régression linéaire (MCO) from scratch

This project implements a linear regression framework in Python developed from scratch to understand the statistical and computational foundations of Ordinary Least Squares (OLS), rather than relying on existing machine-learning libraries.

The program is organized into three main components. The DataSet class manages the data pipeline: it stores the feature matrix and target variable, adds an intercept term, and performs an 80/20 train-test split using pandas DataFrames. The LinearRegression class estimates model coefficients using the analytical OLS solution, generates predictions on new data, and detects issues such as multicollinearity or non-invertible matrices. The Result class evaluates model performance by computing metrics including R², Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

Unit tests were implemented to validate each step of the workflow, from data preparation to prediction accuracy. The model was applied to several economic use cases, including estimation of a risk-aversion parameter, housing price modeling, and analysis of gender wage differences.

Overall, the project demonstrates the ability to translate econometric theory into reliable Python code and to design and evaluate a complete statistical modeling pipeline.
