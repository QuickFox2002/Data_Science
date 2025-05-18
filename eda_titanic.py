
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Basic Information
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Descriptive Statistics
print("\nSummary Statistics:\n", df.describe(include='all'))

# Value counts for categorical variables
print("\nSex Value Counts:\n", df['sex'].value_counts())

# Visualization
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")
plt.show()

sns.histplot(df['age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
