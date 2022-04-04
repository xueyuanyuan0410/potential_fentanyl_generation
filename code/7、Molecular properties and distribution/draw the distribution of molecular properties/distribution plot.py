import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='white',)
import warnings
warnings.filterwarnings('ignore')



fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)

ax1.set_xlim([0,12])

c1, c2, c3 = sns.color_palette('Set1', 3)
df_1 =  pd.read_excel("C:/Users/A/Desktop/fentanyl analogues generation/code/7、Molecular properties and distribution/draw the distribution of molecular properties/fentanyl logp.xlsx")
data_1=df_1['logp']

df_2 =  pd.read_excel("C:/Users/A/Desktop/fentanyl analogues generation/code/7、Molecular properties and distribution/draw the distribution of molecular properties/546 molecular logp.xlsx")
data_2=df_2['logp']

sns.kdeplot(data_1, shade=True, color=c1,label='Fentanyl logp hybridization ratio distribution', ax=ax1)
sns.kdeplot(data_2, shade=True, color=c2, label='10495 valid molecules logp hybridization ratio distribution', ax=ax1)
plt.show()
