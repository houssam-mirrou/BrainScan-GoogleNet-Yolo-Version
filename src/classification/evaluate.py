from src.classification.data import load


def evaluate(checkpoint, output):
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    dataset = load('val')
    truth = np.concatenate([y.numpy() for _, y in dataset])
    prediction = tf.keras.models.load_model(checkpoint, compile=False).predict(dataset).argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=range(4), average='macro', zero_division=0)
    np.savetxt(output / 'confusion_matrix.csv', confusion_matrix(truth, prediction, labels=range(4)), fmt='%d', delimiter=',')
    return dict(accuracy=float(accuracy_score(truth, prediction)), precision=float(precision), recall=float(recall), macro_f1=float(f1))
