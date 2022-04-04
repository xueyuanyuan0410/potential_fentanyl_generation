import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

fig = plt.figure()
df = pd.read_excel('C:/Users/A/Desktop/fentanyl analogues generation/code/6、Molecular similarity/draw molecular similarity heatmap/similarity matrix.xlsx')
sns.heatmap(df, xticklabels=False, yticklabels=False)
plt.show()