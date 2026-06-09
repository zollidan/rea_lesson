import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def main():
    sns.set(style="ticks")

    dataset_path = kagglehub.dataset_download(
        "fivethirtyeight/fivethirtyeight-comic-characters-dataset"
    )

    file_path = dataset_path + "/marvel-wikia-data.csv"

    data = pd.read_csv(
        file_path,
        encoding="latin1",
        sep=","
    )

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

    # Первичная проверка пропусков
    print("\nКолонки с пропущенными значениями:")
    missing_cols = data.columns[data.isnull().sum() > 0]

    for col in missing_cols:
        null_count = data[col].isnull().sum()
        null_percent = round((null_count / total_count) * 100, 2)
        print("{}: {} пропусков, {}%".format(col, null_count, null_percent))

    # Числовые колонки с пропусками
    num_cols = []

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

    # Гистограмма Year
    plt.figure(figsize=(8, 5))

    plt.hist(data["Year"].dropna(), bins=50)

    plt.xlabel("Year")
    plt.ylabel("Количество персонажей")
    plt.title("Распределение персонажей по году первого появления")

    plt.show()

    # Гистограмма APPEARANCES без экстремальных выбросов
    plt.figure(figsize=(8, 5))

    appearances_data = data["APPEARANCES"].dropna()

    x_max = appearances_data.quantile(0.95)
    appearances_filtered = appearances_data[appearances_data <= x_max]

    plt.hist(appearances_filtered, bins=30)

    plt.xlabel("Количество появлений")
    plt.ylabel("Количество персонажей")
    plt.title("Распределение персонажей по количеству появлений")

    plt.show()

    # Обработка числового признака Year
    num_feature = "Year"

    print("\nКоличество пропусков в Year до обработки:")
    print(data[num_feature].isnull().sum())

    # Заполнение пропусков медианой
    imp_num = SimpleImputer(strategy="median")
    data[num_feature + "_imputed"] = imp_num.fit_transform(data[[num_feature]])

    print("\nКоличество пропусков в Year после обработки:")
    print(data[num_feature + "_imputed"].isnull().sum())

    print("\nПервые значения Year после заполнения медианой:")
    print(data[[num_feature, num_feature + "_imputed"]].head())

    # Категориальные колонки с пропусками
    cat_cols = []

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

    # Обработка категориального признака ALIGN
    cat_feature = "ALIGN"

    print("\nУникальные значения ALIGN до обработки:")
    print(data[cat_feature].unique())

    print("\nКоличество пропусков в ALIGN до обработки:")
    print(data[cat_feature].isnull().sum())

    # Заполнение пропусков отдельной категорией Unknown
    imp_cat = SimpleImputer(
        missing_values=np.nan,
        strategy="constant",
        fill_value="Unknown"
    )

    data[cat_feature + "_imputed"] = imp_cat.fit_transform(data[[cat_feature]]).ravel()

    print("\nУникальные значения ALIGN после обработки:")
    print(data[cat_feature + "_imputed"].unique())

    print("\nКоличество пропусков в ALIGN после обработки:")
    print(data[cat_feature + "_imputed"].isnull().sum())

    print("\nКоличество значений Unknown:")
    print((data[cat_feature + "_imputed"] == "Unknown").sum())

    # One-Hot Encoding для ALIGN
    cat_enc = pd.DataFrame({
        "ALIGN": data[cat_feature + "_imputed"]
    })

    cat_dummies = pd.get_dummies(cat_enc)

    print("\nOne-Hot Encoding для ALIGN:")
    print(cat_dummies.head())

    print("\nРазмер после One-Hot Encoding:")
    print(cat_dummies.shape)

    # Итоговый датасет
    data_result = pd.concat(
        [
            data[["name", num_feature + "_imputed", cat_feature + "_imputed"]],
            cat_dummies
        ],
        axis=1
    )

    print("\nИтоговый датасет с обработанными признаками:")
    print(data_result.head())

    print("\nПроверка пропусков в итоговом датасете:")
    print(data_result.isnull().sum())

    # Масштабирование числового признака Year
    scaler = StandardScaler()
    data_result[num_feature + "_scaled"] = scaler.fit_transform(
        data_result[[num_feature + "_imputed"]]
    )

    print("\nМасштабированный числовой признак:")
    print(data_result[[num_feature + "_imputed", num_feature + "_scaled"]].head())


if __name__ == "__main__":
    main()