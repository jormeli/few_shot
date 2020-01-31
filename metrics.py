import gin
import numpy as np


class Metric:
    """Base class for all metrics."""

    def __call__(self, y_preds, y_trues):
        """
        Args:
            y_preds (numpy.ndarray): model predictions
            y_trues (numpy.ndarray): ground truths
        """
        raise NotImplementedError()

    @property
    def name(self):
        """Name of the metric."""
        raise NotImplementedError()


@gin.configurable
class TopKAccuracy(Metric):
    """Calculates top-k accuracy, i.e. whether the ground truth is
    in top-k most confident predictions.

    Args:
        k (int)
    """
    def __init__(self, k=1):
        self.k = k

    @property
    def name(self):
        return 'Top-{}-accuracy'.format(self.k)

    def __call__(self, y_preds, y_trues):
        top_k = np.argsort(y_preds, axis=-1)[:,-self.k:]
        return np.any(top_k.T == y_trues, axis=0).sum()/y_preds.shape[0]


@gin.configurable
class ConfusionMatrix(Metric):
    @property
    def name(self):
        return 'confusion_matrix'

    def __call__(self, y_preds, y_trues):
        n_classes = y_preds.shape[-1]
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        pred_classes = y_preds.argmax(-1)

        for y, pred in zip(y_trues.reshape((-1,)), pred_classes.reshape((-1,))):
            confusion_matrix[y, pred] += 1

        return confusion_matrix


@gin.configurable
class PerClassAccuracy(Metric):
    @property
    def name(self):
        return 'Per-class-accuracy'

    def __call__(self, y_preds, y_trues):
        n_classes = y_preds.shape[-1]
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        pred_classes = y_preds.argmax(-1)

        for y, pred in zip(y_trues.reshape((-1,)), pred_classes.reshape((-1,))):
            confusion_matrix[y, pred] += 1

        return np.diag(confusion_matrix) / confusion_matrix.sum(1)
