import pandas
import matplotlib.pyplot as plt
from pathlib import Path
df1 = pandas.read_csv(Path('25pts100epochs.csv'))
df2 = pandas.read_csv(Path('250pts10epochs.csv'))
df3 = pandas.read_csv(Path('2500pts1epoch.csv'))



fig, axes = plt.subplots(1, 3, figsize=(20, 5))
df1 = df1.drop(axis=1, columns=["Epoch"])
df2 = df2.drop(axis=1, columns=["Epoch"])
df3 = df3.drop(axis=1, columns=["Batch"])
df1.plot(ax=axes[0])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("25 Data Points for 100 Epochs")
df2.plot(ax=axes[1])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("250 Data Points for 10 Epochs")
df3.plot(ax=axes[2])
axes[2].set_xlabel("Batch")
axes[2].set_ylabel("Loss")
axes[2].set_title("2500 Data Points for 1 Epoch")

plt.show()
