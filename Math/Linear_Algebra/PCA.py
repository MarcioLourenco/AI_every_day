### implementation of PCA algorithm 

import numpy as np 

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate correlated data
age = np.random.randint(18, 70, num_samples)
annual_income = age * 500 + np.random.normal(20000, 5000, num_samples)
spending_score = np.random.uniform(1, 100, num_samples)
savings_percentage = np.clip(10 + (70 - age) / 2 + np.random.normal(0, 5, num_samples), 0, 100)
num_purchases = spending_score / 10 + np.random.normal(5, 2, num_samples)

# Create DataFrame
data = pd.DataFrame({
    "Age": age,
    "Annual Income": annual_income,
    "Spending Score": spending_score,
    "Savings Percentage": savings_percentage,
    "Number of Purchases": num_purchases
})

data.to_csv('PCA_data.csv', index=False)