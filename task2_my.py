import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")

# 1. Загрузка данных
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

data = pd.read_csv(f"{path}/heart.csv")

print("\nПервые строки:")
print(data.head())

print("\nРазмер датасета:")
print(data.shape)

print("\nТипы данных:")
print(data.dtypes)

print("\nПропуски:")
print(data.isnull().sum())

print("\nСтатистика:")
print(data.describe())

print("\nУникальные значения target:")
print(data["target"].unique())

sns.boxplot(x=data["age"])
plt.title("Boxplot")
plt.tight_layout()
plt.show()

sns.boxplot(y=data["cp"])
plt.title("Boxplot признака Humidity по вертикали")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data["age"], bins=20, kde=True)
plt.title("Распределение возраста пациентов")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="age", y="thalach", hue="target", data=data)
plt.title("Связь возраста и максимального пульса")
plt.xlabel("Возраст")
plt.ylabel("Максимальная частота пульса")
plt.tight_layout()
plt.show()

corr = data.corr(numeric_only=True)

target_col = "target"

print("\nКорреляция признаков:")
print(corr)

print("\nКорреляция с целевым признаком target:")
target_corr = corr[target_col].sort_values(ascending=False)
print(target_corr)

plt.figure(figsize=(8, 6))
sns.barplot(
    x=target_corr.drop(target_col).values,
    y=target_corr.drop(target_col).index
)
plt.title("Корреляция признаков с target")
plt.xlabel("Коэффициент корреляции")
plt.ylabel("Признак")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.show()

strong_features = target_corr.drop(target_col).abs().sort_values(ascending=False)

