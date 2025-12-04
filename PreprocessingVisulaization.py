
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

df.head()

df.tail()

df.shape

df.drop('id', axis=1, inplace=True)
df.drop('Unnamed: 32', axis=1, inplace=True)

df.shape

df.describe().T

df.info(verbose=True)

df.diagnosis.unique()

df['diagnosis'].value_counts()

df.isnull().sum()

df['diagnosis'] = pd.get_dummies(df['diagnosis'],drop_first=True,dtype=int)

df.head()

df.groupby('diagnosis').mean()

# Set figure size
plt.figure(figsize=(8,6))

# Add a main title
plt.title("Plot of texture_mean vs. radius_mean\n",fontsize=20, fontstyle='italic')

# X- and Y-label with fontsize
plt.xlabel("texture_mean",fontsize=16)
plt.ylabel("radius_mean",fontsize=16)

# # Turn on grid
plt.grid (True)

# # Set Y-axis limit
plt.ylim(df['radius_mean'].min() - 1, df['radius_mean'].max() + 1)

# # X- and Y-axis ticks customization with fontsize and placement
plt.xticks([i*5 for i in range(12)],fontsize=15)
plt.yticks(fontsize=15)

# Main plotting function with choice of color, marker size, and marker edge color
plt.scatter(x=df['texture_mean'],y=df['radius_mean'],c='#d63384',s=150,edgecolors='k')

# # Adding a legend
plt.legend(['radius_mean'],loc=2,fontsize=14)

# Final show method
plt.show()

y = df['diagnosis']

palette_colors = {1:'#74c0fc', 0:'#d63384'}  # 1=Malignant, 0=Benign

plt.figure(figsize=(6,5))
ax = sns.countplot(x=y, hue=y, palette=palette_colors)


B, M = y.value_counts()
print('Number of Benign Tumors:', B)
print('Number of Malignant Tumors:', M)


plt.title('Number of Tumors by Diagnosis', fontsize=16, fontweight='bold')
plt.xlabel('Diagnosis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Diagnosis', labels=['Benign (0)','Malignant (1)'])
plt.grid(alpha=0.3, axis='y')

plt.show()

plt.hist(df['diagnosis'], color='#d63384')
plt.title('Plot_Diagnosis (M=1 , B=0)')
plt.show()

days = np.arange(1, len(df)+1)
candidate_A = df[df['diagnosis']==1]['radius_mean'].values
candidate_B = df[df['diagnosis']==0]['radius_mean'].values

plt.figure(figsize=(12,5))
plt.plot(candidate_A,'o-',markersize=5,c='#d63384',lw=2)
plt.plot(candidate_B,'^-',markersize=5,c='#74c0fc',lw=2)
plt.title('Radius Mean across Samples\n', fontsize=16)
plt.xlabel('Sample Index')
plt.ylabel('Radius Mean')
plt.legend(['Malignant','Benign'])
plt.show()

plt.style.use('ggplot')

malignant = df[df['diagnosis']==1]['radius_mean'].values
benign = df[df['diagnosis']==0]['radius_mean'].values

plt.figure(figsize=(8,6))

box = plt.boxplot([malignant, benign], showmeans=True, patch_artist=True)


colors = ['#d63384', '#74c0fc']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

for median in box['medians']:
    median.set(color='black', linewidth=2)

for mean in box['means']:
    mean.set(marker='o', markerfacecolor='yellow', markeredgecolor='black', markersize=8)

plt.grid(alpha=0.3)
plt.xticks([1,2], ['Malignant (1)', 'Benign (0)'], fontsize=14)
plt.title('Boxplot of Radius Mean by Diagnosis', fontsize=16, fontweight='bold')
plt.ylabel('Radius Mean', fontsize=14)

plt.show()

plt.figure(figsize=(8,6))
sns.regplot(x='radius_mean', y='texture_mean', data=df,
            color='#d63384',
            scatter_kws={'s':50, 'alpha':0.7})

plt.title('Regression: Radius Mean vs Texture Mean', fontsize=16, fontweight='bold')
plt.xlabel('Radius Mean', fontsize=14)
plt.ylabel('Texture Mean', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

sns.lmplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df, height=6, aspect=1.4, markers=['o','^'], palette=['#d63384','#74c0fc'])
plt.title('Regression plot of Radius Mean vs Texture Mean', fontsize=16)
plt.show()

corr_mat = np.corrcoef(df, rowvar=False)
corr_df = pd.DataFrame(corr_mat, columns=df.columns, index=df.columns)
print(np.round(corr_df, 3))

plt.figure(figsize=(20,20))

sns.heatmap(corr_df,
            annot=True,
            fmt=".2f",
            cmap='RdPu',
            linewidths=0.5,
            linecolor='gray')

plt.title('Correlation Matrix Heatmap', fontsize=20, fontweight='bold')
plt.show()

palette_colors = {1:'#d63384', 0:'#74c0fc'}
sns.pairplot(df.iloc[:,0:7], hue='diagnosis', palette=palette_colors, diag_kind='kde')
plt.show()

plt.figure(figsize=(8,5))
df['radius_mean'].plot.density(color='#d63384')
plt.title('Density Plot of Radius Mean', fontsize=16, fontweight='bold')
plt.xlabel('Radius Mean', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

df.to_csv("cleaned_data.csv",index=False)