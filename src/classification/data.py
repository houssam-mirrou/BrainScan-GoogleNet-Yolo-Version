from src.common.validation import ROOT, CLASSES


def load(split):
    import tensorflow as tf
    return tf.keras.utils.image_dataset_from_directory(
        ROOT / 'data/processed/classification' / split,
        class_names=CLASSES, label_mode='int', color_mode='rgb',
        image_size=(224, 224), batch_size=16, shuffle=split == 'train', seed=42,
    ).prefetch(tf.data.AUTOTUNE)  # Saved model already normalizes 0–255 pixels.
