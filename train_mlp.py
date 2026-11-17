"""
train_mlp.py
------------
Trains the small MLP:  10 -> 32 (ReLU) -> 16 (ReLU) -> 4 (softmax)

    python train_mlp.py
"""

import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocess import prepare_data, CLASSES, FEATURES, MODELS, REPORTS, SEED

BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 0.001
VAL_FRACTION = 0.20      # taken from the TRAINING set, the test set is never touched here


def build_model(n_features, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    data = prepare_data(verbose=True, save=True)
    X_train, y_train = data["X_train"], data["y_train"]

    # stratified validation split so every class shows up in validation too
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRACTION, stratify=y_train, random_state=SEED)
    print(f"\nFit: {len(X_fit)}  Validation: {len(X_val)}")

    model = build_model(len(FEATURES), len(CLASSES))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)

    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    model.save(MODELS / "helmet_mlp.keras")

    h = history.history
    best = int(np.argmin(h["val_loss"]))
    print("\n--- Training summary ---")
    print(f"Epochs run            : {len(h['loss'])} (best epoch {best + 1})")
    print(f"Train accuracy (best) : {h['accuracy'][best]:.4f}")
    print(f"Val accuracy   (best) : {h['val_accuracy'][best]:.4f}")
    print(f"Train loss     (best) : {h['loss'][best]:.4f}")
    print(f"Val loss       (best) : {h['val_loss'][best]:.4f}")

    (REPORTS / "training_history.json").write_text(json.dumps(h, indent=2))

    epochs = range(1, len(h["loss"]) + 1)
    for key, title, fname in [("accuracy", "Accuracy vs Epoch", "accuracy_curve.png"),
                              ("loss", "Loss vs Epoch", "loss_curve.png")]:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, h[key], label=f"train {key}")
        plt.plot(epochs, h["val_" + key], label=f"validation {key}")
        plt.axvline(best + 1, color="grey", ls="--", lw=0.8, label="best epoch")
        plt.xlabel("Epoch")
        plt.ylabel(key.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORTS / fname, dpi=130)
        plt.close()
    print("Saved model to models/helmet_mlp.keras and plots to reports/")


if __name__ == "__main__":
    main()
