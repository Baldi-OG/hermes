import logging
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def plot_and_log_confusion_matrix(y_true, y_pred):
    logger.info("Generating Confusion Matrix...")
    labels = sorted(list(set(y_true + y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical", colorbar=False)
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_file="author_confusion.png")
    plt.close(fig)
