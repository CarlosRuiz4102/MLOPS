import numpy as np
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import callbacks, layers, models


def build_mlp(input_dim: int, n_classes: int = 5):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn1d(input_shape, n_classes: int = 5):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dummy(data: dict, seed: int) -> np.ndarray:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(data["X_train"], data["y_train"])
    return model.predict(data["X_val"])


def train_logreg(data: dict, seed: int) -> np.ndarray:
    model = LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(data["X_train_scaled"], data["y_train"])
    return model.predict(data["X_val_scaled"])


def train_random_forest(data: dict, seed: int, fast_run: bool = False):
    model = RandomForestClassifier(
        n_estimators=300 if not fast_run else 30,
        min_samples_leaf=2,
        n_jobs=1,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    model.fit(data["X_train_feat"], data["y_train"])
    return model, model.predict(data["X_val_feat"])


def train_mlp(data: dict, seed: int, fast_run: bool = False):
    tf.random.set_seed(seed)
    batch_size = 256 if not fast_run else 64
    epochs = 30 if not fast_run else 3

    model = build_mlp(data["X_train_scaled"].shape[1])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        data["X_train_scaled"],
        data["y_train"],
        validation_data=(data["X_val_scaled"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=data["class_weight_dict"],
        verbose=0,
        callbacks=[early_stop],
    )
    preds = np.argmax(model.predict(data["X_val_scaled"], verbose=0), axis=1)
    return model, preds


def train_cnn1d(data: dict, seed: int, fast_run: bool = False):
    tf.random.set_seed(seed)
    batch_size = 256 if not fast_run else 64
    epochs = 30 if not fast_run else 3

    model = build_cnn1d((data["X_train_cnn"].shape[1], 1))
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(
        data["X_train_cnn"],
        data["y_train"],
        validation_data=(data["X_val_cnn"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=data["class_weight_dict"],
        verbose=0,
        callbacks=[early_stop],
    )
    preds = np.argmax(model.predict(data["X_val_cnn"], verbose=0), axis=1)
    return model, preds
