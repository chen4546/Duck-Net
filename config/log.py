import datetime
import os
from keras.callbacks import Callback
def logs():
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    log_path = 'logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    os.mkdir(log_path + formatted_date)
    os.mkdir(os.path.join(log_path + formatted_date,'train_log'))
    return log_path+formatted_date

class EpochLossLog(Callback):
    def __init__(self,file_path,model_path):
        super().__init__()
        self.file_path = file_path
        self.file = None
        self.best_val_loss = None
        self.model_path=model_path

    def on_train_begin(self, logs=None):
        self.file = open(self.file_path, 'w', encoding='utf-8')

    def on_epoch_end(self, epoch, logs=None):
        self.file.write(f"{logs['loss']}\n")
        self.file.flush()
        if logs is not None:
            print(f"Total Loss: {logs.get('loss'):.3f}")
        val_loss = logs.get('val_loss')
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"Saving best model to {self.model_path}")

    def on_train_end(self, logs=None):
        self.file.close()
