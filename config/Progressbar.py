import tensorflow as tf
from tqdm import tqdm
class EpochProcessBar(tf.keras.callbacks.Callback):
    def __init__(self,total_epochs):
        super().__init__()
        self.total_epochs=total_epochs
        self.epoch_bar=None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_pbar = tqdm(
            total=self.params['steps'],
            desc=f"Epoch {epoch + 1}/{self.total_epochs}",
            unit='batch',
            dynamic_ncols=True
        )

    def on_train_batch_end(self, batch, logs=None):
        # 更新进度条并显示当前指标
        self.epoch_pbar.update(1)
        self.epoch_pbar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'accuracy': f"{logs['accuracy']:.4f}"
        })

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_pbar.close()