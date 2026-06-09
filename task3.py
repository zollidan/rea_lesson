import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sns.set(style="ticks")

# Загружаем исходный датасет с Kaggle и формируем путь до CSV-файла.
dataset_path = kagglehub.dataset_download(
    "fivethirtyeight/fivethirtyeight-comic-characters-dataset"
)

file_path = dataset_path + "/marvel-wikia-data.csv"

data = pd.read_csv(
    file_path,
    encoding="latin1",
    sep=","
)

# Базовый первичный осмотр: размер, типы, пропуски и первые строки.
print("Размер датасета:")
print(data.shape)

print("\nТипы данных:")
print(data.dtypes)

print("\nКоличество пропусков:")
print(data.isnull().sum())

print("\nПервые 5 строк:")
print(data.head())

total_count = data.shape[0]
print("\nВсего строк: {}".format(total_count))

# Показываем, в каких колонках есть пропуски и насколько они велики в процентах.
print("\nКолонки с пропущенными значениями:")
missing_cols = data.columns[data.isnull().sum() > 0]

for col in missing_cols:
    null_count = data[col].isnull().sum()
    null_percent = round((null_count / total_count) * 100, 2)
    print("{}: {} пропусков, {}%".format(col, null_count, null_percent))

# Демонстрация трех типовых подходов к пропускам:
# 1) удалить проблемные колонки,
# 2) удалить строки с пропусками,
# 3) заменить пропуски значением по умолчанию.
data_new_1 = data.dropna(axis=1, how="any")
data_new_2 = data.dropna(axis=0, how="any")
data_new_3 = data.fillna(0)

print("\nУдаление колонок с пропусками:")
print(data.shape, data_new_1.shape)

print("\nУдаление строк с пропусками:")
print(data.shape, data_new_2.shape)

print("\nЗаполнение всех пропусков нулями:")
print(data_new_3.head())

num_cols = []

# Ищем только числовые признаки, где есть пропуски.
print("\nЧисловые колонки с пропусками:")
for col in data.columns:
    temp_null_count = data[col].isnull().sum()
    dt = str(data[col].dtype)

    if temp_null_count > 0 and (dt == "float64" or dt == "int64"):
        num_cols.append(col)
        temp_perc = round((temp_null_count / total_count) * 100.0, 2)
        print("Колонка {}. Тип данных {}. Количество пустых значений {}, {}%.".format(
            col, dt, temp_null_count, temp_perc
        ))

data_num = data[num_cols]
print("\nЧисловые признаки с пропусками:")
print(data_num.head())

# Для числовых признаков строим гистограммы, чтобы понять распределение
# и выбрать адекватную стратегию заполнения пропусков.
for col in data_num:
    plt.figure(figsize=(8, 5))
    plt.hist(data[col].dropna(), 50)
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.title("Гистограмма признака {}".format(col))
    plt.show()

num_feature = "Year"

# Ниже точечно разбираем один числовой признак Year:
# смотрим, где были пропуски, и сравниваем несколько стратегий импьютации.
data_num_year = data[[num_feature]]

indicator = MissingIndicator()
mask_missing_values_only = indicator.fit_transform(data_num_year)

strategies = ["mean", "median", "most_frequent"]

print("\nИмпьютация числового признака Year:")

for strategy in strategies:
    imp_num = SimpleImputer(strategy    =strategy)
    data_num_imp = imp_num.fit_transform(data_num_year)
    print("\nСтратегия:", strategy)
    print("Значения, которыми были заполнены пропуски:")
    print(data_num_imp[mask_missing_values_only][:10])

# В итоговый датасет сохраняем версию Year, заполненную медианой.
imp_num_final = SimpleImputer(strategy="median")
data[num_feature + "_imputed"] = imp_num_final.fit_transform(data[[num_feature]])

print("\nПроверка пропусков после импьютации Year:")
print(data[num_feature + "_imputed"].isnull().sum())

cat_cols = []

# Аналогично ищем категориальные признаки с пропусками.
print("\nКатегориальные колонки с пропусками:")
for col in data.columns:
    temp_null_count = data[col].isnull().sum()
    dt = str(data[col].dtype)

    if temp_null_count > 0 and dt == "object":
        cat_cols.append(col)
        temp_perc = round((temp_null_count / total_count) * 100.0, 2)
        print("Колонка {}. Тип данных {}. Количество пустых значений {}, {}%.".format(
            col, dt, temp_null_count, temp_perc
        ))

cat_feature = "ALIGN"

# Для примера берем категориальный признак ALIGN и смотрим его значения до обработки.
cat_temp_data = data[[cat_feature]]

print("\nУникальные значения категориального признака до обработки:")
print(cat_temp_data[cat_feature].unique())

print("\nКоличество пропусков в ALIGN:")
print(cat_temp_data[cat_feature].isnull().sum())

# Сравниваем два популярных способа заполнения категориальных пропусков:
# наиболее частым значением и специальной меткой Unknown.
imp_cat_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
data_imp_most_frequent = imp_cat_most_frequent.fit_transform(cat_temp_data)

print("\nИмпьютация ALIGN наиболее частым значением:")
print(np.unique(data_imp_most_frequent))

imp_cat_constant = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Unknown")
data_imp_constant = imp_cat_constant.fit_transform(cat_temp_data)

print("\nИмпьютация ALIGN константой Unknown:")
print(np.unique(data_imp_constant))

print("\nКоличество значений Unknown:")
print((data_imp_constant == "Unknown").sum())

# Для дальнейшей обработки оставляем вариант с явной меткой Unknown.
data[cat_feature + "_imputed"] = data_imp_constant.ravel()

cat_enc = pd.DataFrame({
    "ALIGN": data[cat_feature + "_imputed"]
})

print("\nДанные для кодирования:")
print(cat_enc.head())

# LabelEncoder заменяет каждую категорию одним числом.
# Это компактно, но добавляет искусственный порядок между классами.
le = LabelEncoder()
cat_enc_le = le.fit_transform(cat_enc["ALIGN"])

print("\nКлассы LabelEncoder:")
print(le.classes_)

print("\nПервые 10 значений после Label Encoding:")
print(cat_enc_le[:10])

print("\nУникальные значения после Label Encoding:")
print(np.unique(cat_enc_le))

data[cat_feature + "_label_encoded"] = cat_enc_le

# One-Hot Encoding создает отдельный бинарный столбец под каждую категорию.
ohe = OneHotEncoder()
cat_enc_ohe = ohe.fit_transform(cat_enc[["ALIGN"]])

print("\nРазмер до One-Hot Encoding:")
print(cat_enc.shape)

print("\nРазмер после One-Hot Encoding:")
print(cat_enc_ohe.shape)

print("\nПервые 10 строк One-Hot Encoding:")
print(cat_enc_ohe.todense()[0:10])

cat_dummies = pd.get_dummies(cat_enc)

print("\nOne-Hot Encoding через pandas get_dummies:")
print(cat_dummies.head())

# Собираем компактный итоговый набор:
# имя персонажа, обработанный числовой признак и обе версии кодирования категории.
data_result = pd.concat(
    [
        data[["name", num_feature + "_imputed", cat_feature + "_imputed", cat_feature + "_label_encoded"]],
        cat_dummies
    ],
    axis=1
)

print("\nИтоговый датасет с обработанными признаками:")
print(data_result.head())

print("\nПроверка пропусков в итоговом датасете:")
print(data_result.isnull().sum())

# Масштабируем числовой признак, чтобы привести его к стандартному виду
# со средним около 0 и стандартным отклонением около 1.
scaler = StandardScaler()
data_result[num_feature + "_scaled"] = scaler.fit_transform(data_result[[num_feature + "_imputed"]])

print("\nМасштабированный числовой признак:")
print(data_result[[num_feature + "_imputed", num_feature + "_scaled"]].head())
