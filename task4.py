import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.set_printoptions(precision=3, suppress=True)
sns.set(style="ticks")

# Проверяем версию TensorFlow, с которой запускается скрипт.
print(tf.__version__)

# Загружаем обучающий датасет по ценам домов.
data = pd.read_csv("train.csv", sep=",", na_values=["NA", ""])

# Базовый обзор данных: размер, первые строки, типы и пропуски.
print("Размер датасета:")
print(data.shape)

print("\nПервые 5 строк:")
print(data.head())

print("\nТипы данных:")
print(data.dtypes)

print("\nКоличество пропусков:")
print(data.isnull().sum().sort_values(ascending=False).head(20))

print("\nОписание SalePrice:")
print(data["SalePrice"].describe())

# Отдельно смотрим распределение целевой переменной SalePrice.
plt.figure(figsize=(8, 5))
plt.hist(data["SalePrice"], bins=30)
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.title("Распределение SalePrice")
plt.show()

num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

# Целевую переменную не включаем в список признаков для заполнения и масштабирования.
num_cols.remove("SalePrice")

# Числовые пропуски заменяем медианой по столбцу.
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Категориальные пропуски заполняем отдельной меткой Unknown.
for col in cat_cols:
    data[col] = data[col].fillna("Unknown")

print("\nПропуски после обработки:")
print(data.isnull().sum().sum())

# Преобразуем категориальные признаки в набор бинарных столбцов.
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

print("\nРазмер после кодирования:")
print(data_encoded.shape)

# Разделяем датасет на признаки X и целевую переменную y.
X = data_encoded.drop("SalePrice", axis=1)
y = data_encoded["SalePrice"]

# Делим данные на обучающую и тестовую части для честной оценки модели.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

# Стандартизируем признаки: обучаем scaler только на train и применяем к test.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nРазмер обучающей выборки:")
print(X_train_scaled.shape)

print("\nРазмер тестовой выборки:")
print(X_test_scaled.shape)

# Слой нормализации Keras дополнительно сохраняет статистики признаков
# и используется как первый слой в обеих нейросетях.
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train_scaled))

print("\nСредние значения слоя нормализации:")
print(normalizer.mean.numpy())

# Функция для визуального контроля обучения:
# сравнивает ошибку на обучении и валидации по эпохам.
def plot_loss(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Унифицированная оценка модели на тестовой выборке по основным метрикам регрессии.
def evaluate_model(model, X_test_data, y_test_data, model_name):
    predictions = model.predict(X_test_data).flatten()

    mae = mean_absolute_error(y_test_data, predictions)
    mse = mean_squared_error(y_test_data, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_data, predictions)

    print("\n", model_name)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)

    return mae, mse, rmse, r2, predictions

# Базовая линейная модель: фактически один полносвязный слой без скрытых слоев.
linear_model = keras.Sequential([
    normalizer,
    layers.Dense(1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss="mean_absolute_error"
)

print("\nОднослойная модель:")
linear_model.summary()

# Обучаем модель и одновременно следим за качеством на валидационной части train.
history_linear = linear_model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

plot_loss(history_linear, "Ошибка однослойной модели")

test_results = {}

# Сохраняем метрики первой модели для дальнейшего сравнения.
mae, mse, rmse, r2, linear_predictions = evaluate_model(
    linear_model,
    X_test_scaled,
    y_test,
    "Однослойная нейронная сеть"
)

test_results["linear_model"] = [mae, mse, rmse, r2]

# Более сложная многослойная сеть с несколькими скрытыми слоями ReLU.
dnn_model = keras.Sequential([
    normalizer,
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

dnn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss="mean_absolute_error"
)

print("\nМногослойная модель:")
dnn_model.summary()

# Обучаем вторую модель на тех же подготовленных данных.
history_dnn = dnn_model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

plot_loss(history_dnn, "Ошибка многослойной модели")

mae, mse, rmse, r2, dnn_predictions = evaluate_model(
    dnn_model,
    X_test_scaled,
    y_test,
    "Многослойная нейронная сеть"
)

test_results["dnn_model"] = [mae, mse, rmse, r2]

# Собираем итоговую таблицу по всем моделям.
results_df = pd.DataFrame(
    test_results,
    index=["MAE", "MSE", "RMSE", "R2"]
).T

print("\nИтоговая таблица качества моделей:")
print(results_df)

# Сравниваем реальные и предсказанные значения для лучшей визуальной оценки качества.
plt.figure(figsize=(7, 7))
plt.scatter(y_test, dnn_predictions)
plt.xlabel("True Values SalePrice")
plt.ylabel("Predictions SalePrice")
plt.title("Истинные и предсказанные значения")
lims = [0, max(y_test.max(), dnn_predictions.max())]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.grid(True)
plt.show()

# Анализируем распределение ошибок предсказания.
error = dnn_predictions - y_test

plt.figure(figsize=(8, 5))
plt.hist(error, bins=25)
plt.xlabel("Prediction Error SalePrice")
plt.ylabel("Count")
plt.title("Распределение ошибок")
plt.show()

# Определяем лучшую модель по минимальному значению MAE.
best_model = results_df["MAE"].idxmin()
