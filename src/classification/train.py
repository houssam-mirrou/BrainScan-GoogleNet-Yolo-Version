"""Continue the explicitly selected frozen V2 checkpoint with notebook settings."""
from src.common.validation import ROOT
from src.classification.data import load

PARAMS = dict(architecture='InceptionV3_MultiScale_V2', epochs=25, batch=16,
              learning_rate=0.001, seed=42, initialization='selected_checkpoint')


def train(output):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(ROOT / 'models/classification/inceptionv3_multiscale_v2.keras', compile=False)
    model.get_layer('inceptionv3_multiscale_backbone').trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = output / 'best.keras'
    model.fit(load('train'), validation_data=load('val'), epochs=25, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(output / 'history.csv'),
    ])
    return checkpoint
