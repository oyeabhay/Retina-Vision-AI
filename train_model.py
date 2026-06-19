"""
Human Eye Disease Prediction System
Train MobileNetV2 on OCT (Optical Coherence Tomography) Dataset

Dataset: https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct
Classes: CNV, DME, DRUSEN, NORMAL

Usage:
    python train_model.py --data_dir /path/to/OCT2017 --epochs 10
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 4
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

DISEASE_INFO = {
    "CNV": "Choroidal Neovascularization – abnormal blood vessel growth beneath the retina.",
    "DME": "Diabetic Macular Edema – fluid accumulation in the macula due to diabetes.",
    "DRUSEN": "Drusen – small yellow deposits under the retina; early sign of AMD.",
    "NORMAL": "Normal – no signs of retinal disease detected.",
}


# ─────────────────────────────────────────────
#  DATA GENERATORS
# ─────────────────────────────────────────────
def build_generators(data_dir: str):
    """
    Create train / val / test ImageDataGenerators.
    Auto-detects whether a separate 'val' folder exists.
    Also handles datasets where test folder is named 'test' or 'val'.
    """

    # ── Resolve sub-folder paths ─────────────
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    # Fallback: some Kaggle downloads put test images under 'val'
    if not os.path.isdir(test_dir) and os.path.isdir(val_dir):
        test_dir = val_dir

    # Validate
    for d, name in [(train_dir, "train"), (test_dir, "test/val")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"\n\n❌ Could not find '{name}' folder inside:\n   {data_dir}\n\n"
                f"Expected structure:\n"
                f"  {data_dir}/\n"
                f"  ├── train/\n"
                f"  │   ├── CNV/\n"
                f"  │   ├── DME/\n"
                f"  │   ├── DRUSEN/\n"
                f"  │   └── NORMAL/\n"
                f"  ├── val/      (optional)\n"
                f"  └── test/\n"
            )

    has_val_folder = os.path.isdir(val_dir) and val_dir != test_dir
    print(f"📁 train dir : {train_dir}")
    print(f"📁 val dir   : {val_dir if has_val_folder else '(using 10% split of train)'}")
    print(f"📁 test dir  : {test_dir}")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    if has_val_folder:
        # ── Use the real val folder ───────────
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
        )

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
        )

        val_gen = test_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

    else:
        # ── No val folder → 10 % split of train ──
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            validation_split=0.1,
        )

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, learning_rate: float = 1e-4) -> Model:
    """
    Transfer learning with MobileNetV2.
    Phase 1 → Train only the custom head (base frozen).
    Phase 2 → Fine-tune top layers of the base (see fine_tune_model).
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False          # freeze during phase-1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_model(model: Model, unfreeze_from: int = 100,
                    learning_rate: float = 1e-5) -> Model:
    """Unfreeze the top layers of MobileNetV2 for fine-tuning (phase 2)."""

    # Find MobileNetV2 base by looking for a layer that has sub-layers
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            base_model = layer
            break

    if base_model is None:
        print("Could not find nested base model — unfreezing top-level layers instead.")
        for layer in model.layers[:-4]:
            layer.trainable = True
        for layer in model.layers[:-4][:unfreeze_from]:
            layer.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:unfreeze_from]:
            layer.trainable = False
        print(f"Fine-tuning base model: {base_model.name}")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Fine-tuning: {sum(l.trainable for l in model.layers)} trainable layers")
    return model


# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(model_path: str, log_path: str):
    return [
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(log_path, append=True),
    ]


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train(data_dir: str, output_dir: str, epochs: int, fine_tune_epochs: int, skip_phase1: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    train_gen, val_gen, test_gen = build_generators(data_dir)

    # Save class-index mapping
    class_indices = train_gen.class_indices          # e.g. {'CNV': 0, ...}
    idx_to_class  = {v: k for k, v in class_indices.items()}
    with open(os.path.join(output_dir, "class_indices.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print("Class mapping:", idx_to_class)

    # ── Phase 1: train head (or skip) ───────
    phase1_path = os.path.join(output_dir, "best_model_phase1.keras")
    history1 = None

    if skip_phase1:
        if not os.path.exists(phase1_path):
            raise FileNotFoundError(
                f"\n❌ Cannot skip Phase 1 — saved model not found at:\n   {phase1_path}\n"
                "Run without --skip_phase1 first."
            )
        print(f"\n⏭️  Skipping Phase 1 — loading model from:\n   {phase1_path}")
        model = tf.keras.models.load_model(phase1_path)
        print("✅ Phase 1 model loaded.")
    else:
        print("\n=== Phase 1: Training custom head (base frozen) ===")
        model = build_model()
        model.summary()

        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=get_callbacks(phase1_path,
                                    os.path.join(output_dir, "log_phase1.csv")),
        )

    # ── Phase 2: fine-tune ───────────────────
    if fine_tune_epochs > 0:
        print("\n=== Phase 2: Fine-tuning top layers ===")
        model = fine_tune_model(model)

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            callbacks=get_callbacks(
                os.path.join(output_dir, "best_model.keras"),
                os.path.join(output_dir, "log_phase2.csv"),
            ),
        )
    else:
        # Use phase-1 best as final model
        model.save(os.path.join(output_dir, "best_model.keras"))

    # ── Evaluation ───────────────────────────
    print("\n=== Evaluating on Test Set ===")
    loss, acc = model.evaluate(test_gen)
    print(f"Test  Loss: {loss:.4f}")
    print(f"Test  Accuracy: {acc:.4f}")

    # Classification report
    preds     = model.predict(test_gen)
    y_pred    = np.argmax(preds, axis=1)
    y_true    = test_gen.classes
    labels    = [idx_to_class[i] for i in range(NUM_CLASSES)]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix – Eye Disease Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {output_dir}/confusion_matrix.png")

    # Training history plots
    if history1:
        _plot_history(history1, output_dir, tag="phase1")
    if fine_tune_epochs > 0:
        _plot_history(history2, output_dir, tag="phase2")

    print(f"\nAll outputs saved to: {output_dir}/")


def _plot_history(history, output_dir: str, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"Accuracy ({tag})")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"Loss ({tag})")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_history_{tag}.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Eye Disease Classifier")
    parser.add_argument("--data_dir",          type=str, default="OCT2017",
                        help="Root of the OCT dataset (contains train/ and test/ folders)")
    parser.add_argument("--output_dir",        type=str, default="model",
                        help="Directory to save the trained model and logs")
    parser.add_argument("--epochs",            type=int, default=10,
                        help="Epochs for phase-1 (head training)")
    parser.add_argument("--fine_tune_epochs",  type=int, default=5,
                        help="Epochs for phase-2 (fine-tuning); set 0 to skip")
    parser.add_argument("--skip_phase1",        action="store_true",
                        help="Skip Phase 1 and load best_model_phase1.keras directly")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        skip_phase1=args.skip_phase1,
    )
