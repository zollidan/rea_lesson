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

print(tf.__version__)

data = pd.read_csv("train.csv", sep=",", na_values=["NA", ""])

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

plt.figure(figsize=(8, 5))
plt.hist(data["SalePrice"], bins=30)
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.title("Распределение SalePrice")
plt.show()

num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

num_cols.remove("SalePrice")

for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

for col in cat_cols:
    data[col] = data[col].fillna("Unknown")

print("\nПропуски после обработки:")
print(data.isnull().sum().sum())

data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

print("\nРазмер после кодирования:")
print(data_encoded.shape)

X = data_encoded.drop("SalePrice", axis=1)
y = data_encoded["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nРазмер обучающей выборки:")
print(X_train_scaled.shape)

print("\nРазмер тестовой выборки:")
print(X_test_scaled.shape)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train_scaled))

print("\nСредние значения слоя нормализации:")
print(normalizer.mean.numpy())

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

history_linear = linear_model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

plot_loss(history_linear, "Ошибка однослойной модели")

test_results = {}

mae, mse, rmse, r2, linear_predictions = evaluate_model(
    linear_model,
    X_test_scaled,
    y_test,
    "Однослойная нейронная сеть"
)

test_results["linear_model"] = [mae, mse, rmse, r2]

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

results_df = pd.DataFrame(
    test_results,
    index=["MAE", "MSE", "RMSE", "R2"]
).T

print("\nИтоговая таблица качества моделей:")
print(results_df)

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

error = dnn_predictions - y_test

plt.figure(figsize=(8, 5))
plt.hist(error, bins=25)
plt.xlabel("Prediction Error SalePrice")
plt.ylabel("Count")
plt.title("Распределение ошибок")
plt.show()

best_model = results_df["MAE"].idxmin()
