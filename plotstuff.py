import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Assuming y_train and x_calculated are your data
df = pd.DataFrame({"Y_Train": y_train, "X_Calculated": x_calculated})

sns.scatterplot(data=df, x="X_Calculated", y="Y_Train")

plt.show()

import numpy as np

# Calculate means and standard deviations
x_mean, x_sigma = df["X_Calculated"].mean(), df["X_Calculated"].std()
y_mean, y_sigma = df["Y_Train"].mean(), df["Y_Train"].std()

# Create scatter plot with trendline
sns.regplot(data=df, x="X_Calculated", y="Y_Train")

# Draw lines at the means
plt.axvline(x_mean, color="r", linestyle="--")
plt.axhline(y_mean, color="r", linestyle="--")

# Highlight the area within one standard deviation of the mean
plt.fill_between(
    [x_mean - x_sigma, x_mean + x_sigma],
    y_mean - y_sigma,
    y_mean + y_sigma,
    color="red",
    alpha=0.2,
)

plt.show()
