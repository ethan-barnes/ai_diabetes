import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_csv('C:/Users/Ethan/Documents/School/Uni/AI/dataset_diabetes/diabetic_data_train_x.csv')

for category in X_train:
    sns.distplot(X_train[category].dropna(), kde=False, color='black', bins=15)
    plt.show()
