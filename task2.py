import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")

data = pd.read_csv("occupancy+detection/datatraining.txt", sep=",")

print("\nПервые 5 строк датасета:")
print(data.head())

print("\nРазмер датасета:")
print(data.shape)

total_count = data.shape[0]
print("\nВсего строк: {}".format(total_count))

print("\nСписок колонок:")
print(data.columns)

print("\nСписок колонок с типами данных:")
print(data.dtypes)

print("\nПроверка наличия пустых значений:")
for col in data.columns:
    temp_null_count = data[data[col].isnull()].shape[0]
    print("{} - {}".format(col, temp_null_count))

print("\nОсновные статистические характеристики набора данных:")
print(data.describe())

print("\nУникальные значения целевого признака Occupancy:")
print(data["Occupancy"].unique())

print("\nЦелевой признак является бинарным и содержит только значения 0 и 1.")

numeric_data = data.drop(columns=["date"])

# Диаграмма рассеяния
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(ax=ax, x="Humidity", y="HumidityRatio", data=data)
plt.title("Диаграмма рассеяния Humidity и HumidityRatio")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(ax=ax, x="Humidity", y="HumidityRatio", data=data, hue="Occupancy")
plt.title("Humidity и HumidityRatio с группировкой по Occupancy")
plt.tight_layout()
plt.show()

# гистограмма
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data["Humidity"], kde=True, ax=ax)
plt.title("Распределение признака Humidity")
plt.tight_layout()
plt.show()

# Jointplot
sns.jointplot(x="Humidity", y="HumidityRatio", data=data)
plt.show()

sns.jointplot(x="Humidity", y="HumidityRatio", data=data, kind="hex")
plt.show()

sns.jointplot(x="Humidity", y="HumidityRatio", data=data, kind="kde")
plt.show()

# Парные диаграммы
sns.pairplot(data)
plt.show()

sns.pairplot(data, hue="Occupancy")
plt.show()

# Ящик с усами
sns.boxplot(x=data["Humidity"])
plt.title("Boxplot признака Humidity по горизонтали")
plt.tight_layout()
plt.show()

sns.boxplot(y=data["Humidity"])
plt.title("Boxplot признака Humidity по вертикали")
plt.tight_layout()
plt.show()

sns.boxplot(x="Occupancy", y="Humidity", data=data)
plt.title("Boxplot Humidity по группам Occupancy")
plt.tight_layout()
plt.show()

# Violin plot
sns.violinplot(x=data["Humidity"])
plt.title("Violin plot признака Humidity")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.violinplot(ax=ax[0], x=data["Humidity"])
sns.histplot(data["Humidity"], kde=True, ax=ax[1])
ax[0].set_title("Violin plot признака Humidity")
ax[1].set_title("Гистограмма признака Humidity")
plt.tight_layout()
plt.show()

sns.violinplot(x="Occupancy", y="Humidity", data=data)
plt.title("Violin plot Humidity по группам Occupancy")
plt.tight_layout()
plt.show()

sns.catplot(y="Humidity", x="Occupancy", data=data, kind="violin")
plt.title("Catplot: Humidity по группам Occupancy")
plt.tight_layout()
plt.show()

# Информация о корреляции признаков
print("\nКорреляционная матрица:")
corr = numeric_data.corr()
print(corr)

print("\nКорреляция признаков с целевым признаком Occupancy:")
target_corr = corr["Occupancy"].sort_values(ascending=False)
print(target_corr)

# Тепловая карта
sns.heatmap(corr)
plt.title("Тепловая карта корреляции")
plt.tight_layout()
plt.show()

# Вывод значений в ячейках
sns.heatmap(corr, annot=True, fmt=".3f")
plt.title("Тепловая карта корреляции со значениями")
plt.tight_layout()
plt.show()

# Изменение цветовой гаммы
sns.heatmap(corr, cmap="YlGnBu", annot=True, fmt=".3f")
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.show()
