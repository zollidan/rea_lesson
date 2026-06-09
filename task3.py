import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

sns.set(style="ticks")


data = pd.read_csv("train.csv", sep=",")

print("Размер набора данных:")
print(data.shape)

print("\nТипы колонок:")
print(data.dtypes)

print("\nКоличество пропущенных значений:")
print(data.isnull().sum())

print("\nПервые 5 строк датасета:")
print(data.head())

total_count = data.shape[0]
print("Всего строк: {}".format(total_count))

data_new_1 = data.dropna(axis=1, how="any")
print("\nРазмер после удаления колонок с пропусками:")
print(data.shape, data_new_1.shape)

# Удаление строк, содержащих пустые значения
data_new_2 = data.dropna(axis=0, how="any")
print("\nРазмер после удаления строк с пропусками:")
print(data.shape, data_new_2.shape)

print("\nПервые строки исходного датасета:")
print(data.head())

data_new_3 = data.fillna(0)

print("\nПервые строки после заполнения пропусков нулями:")
print(data_new_3.head())


num_cols = []

for col in data.columns:
    temp_null_count = data[data[col].isnull()].shape[0]
    dt = str(data[col].dtype)

    if temp_null_count > 0 and (dt == "float64" or dt == "int64"):
        num_cols.append(col)
        temp_perc = round((temp_null_count / total_count) * 100.0, 2)

        print(
            "Колонка {}. Тип данных {}. Количество пустых значений {}, {}%.".format(
                col,
                dt,
                temp_null_count,
                temp_perc
            )
        )

data_num = data[num_cols]

print("\nЧисловые колонки с пропусками:")
print(data_num)


for col in data_num:
    plt.hist(data[col], 50)
    plt.xlabel(col)
    plt.show()


# Импьютация числового признака MasVnrArea
data_num_MasVnrArea = data_num[["MasVnrArea"]]

print("\nПервые строки MasVnrArea:")
print(data_num_MasVnrArea.head())

indicator = MissingIndicator()
mask_missing_values_only = indicator.fit_transform(data_num_MasVnrArea)

print("\nМаска пропущенных значений:")
print(mask_missing_values_only)

strategies = ["mean", "median", "most_frequent"]


def test_num_impute(strategy_param):
    imp_num = SimpleImputer(strategy=strategy_param)
    data_num_imp = imp_num.fit_transform(data_num_MasVnrArea)
    return data_num_imp[mask_missing_values_only]


print("\nИмпьютация средним:")
print(strategies[0], test_num_impute(strategies[0]))

print("\nИмпьютация медианой:")
print(strategies[1], test_num_impute(strategies[1]))

print("\nИмпьютация наиболее частым значением:")
print(strategies[2], test_num_impute(strategies[2]))


# Замена переменных в исходном массиве
imp_mean = SimpleImputer(strategy="mean")
data_num_MasVnrArea1 = imp_mean.fit_transform(data_num_MasVnrArea)

print("\nПроверка заполненных значений:")
print(data_num_MasVnrArea1[mask_missing_values_only])


# Обработка пропусков в категориальных данных
cat_cols = []

for col in data.columns:
    temp_null_count = data[data[col].isnull()].shape[0]
    dt = str(data[col].dtype)

    if temp_null_count > 0 and dt == "object":
        cat_cols.append(col)
        temp_perc = round((temp_null_count / total_count) * 100.0, 2)

        print(
            "Колонка {}. Тип данных {}. Количество пустых значений {}, {}%.".format(
                col,
                dt,
                temp_null_count,
                temp_perc
            )
        )


# Импьютация категориального признака MasVnrType
cat_temp_data = data[["MasVnrType"]]

print("\nПервые строки MasVnrType:")
print(cat_temp_data.head())

print("\nУникальные значения MasVnrType:")
print(cat_temp_data["MasVnrType"].unique())

print("\nКоличество пропусков MasVnrType:")
print(cat_temp_data[cat_temp_data["MasVnrType"].isnull()].shape)


# Импьютация наиболее частыми значениями
imp2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
data_imp2 = imp2.fit_transform(cat_temp_data)

print("\nДанные после импьютации most_frequent:")
print(data_imp2)

print("\nУникальные значения после most_frequent:")
print(np.unique(data_imp2))


# Импьютация константой
imp3 = SimpleImputer(
    missing_values=np.nan,
    strategy="constant",
    fill_value="NA"
)

data_imp3 = imp3.fit_transform(cat_temp_data)

print("\nДанные после импьютации constant:")
print(data_imp3)

print("\nУникальные значения после constant:")
print(np.unique(data_imp3))

print("\nКоличество значений NA:")
print(data_imp3[data_imp3 == "NA"].size)


# Преобразование категориальных признаков в числовые
cat_enc = pd.DataFrame({"c1": data_imp2.T[0]})

print("\nКатегориальный признак после импьютации:")
print(cat_enc)


# Label Encoding
print("\nУникальные значения перед LabelEncoder:")
print(cat_enc["c1"].unique())

le = LabelEncoder()
cat_enc_le = le.fit_transform(cat_enc["c1"])

print("\nКлассы LabelEncoder:")
print(le.classes_)

print("\nРезультат LabelEncoder:")
print(cat_enc_le)

print("\nУникальные значения после LabelEncoder:")
print(np.unique(cat_enc_le))

print("\nОбратное преобразование:")
print(le.inverse_transform([0, 1, 2]))


# One-hot encoding
ohe = OneHotEncoder()
cat_enc_ohe = ohe.fit_transform(cat_enc[["c1"]])

print("\nРазмер исходного категориального признака:")
print(cat_enc.shape)

print("\nРазмер после OneHotEncoder:")
print(cat_enc_ohe.shape)

print("\nРезультат OneHotEncoder:")
print(cat_enc_ohe)

print("\nПервые 10 строк OneHotEncoder в плотном виде:")
print(cat_enc_ohe.todense()[0:10])

print("\nПервые 10 строк исходного признака:")
print(cat_enc.head(10))


print("\nРезультат pd.get_dummies:")
print(pd.get_dummies(cat_enc).head())