import tensorflow as tf
import os
from ImageLoader.ImageLoader2D import load_data
from ModelArchitecture.DUCK_Net import create_model
from ModelArchitecture.DiceLoss import dice_metric_loss
from config.Progressbar import EpochProcessBar
from keras.callbacks import ModelCheckpoint
from config.log import logs,EpochLossLog
from config.epoch_loss import plot_training_curve
if __name__=='__main__':
    log_dir=logs()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    img_shape = [256, 256]
    batch_size = 2
    epochs = 100
    start_filters = 8
    dataset = 'aiot'
    X_train, Y_train = load_data(
        img_height=img_shape[0],
        img_width=img_shape[1],
        images_to_be_loaded=-1,
        dataset=dataset
    )
    split = int(0.8 * len(X_train))
    X_train, X_val = X_train[:split], X_train[split:]
    Y_train, Y_val = Y_train[:split], Y_train[split:]

    model = create_model(
        img_height=img_shape[0],
        img_width=img_shape[1],
        input_chanels=3,
        out_classes=1,
        starting_filters=start_filters
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_metric_loss,
        metrics=['accuracy']
    )
    steps_per_epoch = len(X_train) // batch_size
    callbacks = [
        EpochProcessBar(total_epochs=epochs),
        EpochLossLog(os.path.join(log_dir,'train_log/train_loss.txt'),'best_epoch_model.h5'),
        ModelCheckpoint(
            os.path.join(log_dir, 'best_epoch_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,  # 保存完整模型
            mode='min',
            verbose=0,
            message='111111'
        ),
        ModelCheckpoint(
            os.path.join(log_dir, 'last_epoch_model.h5'),
            save_weights_only=False,
            save_freq='epoch',  # 每个epoch保存一次
            verbose=0
        ),
        ModelCheckpoint(
            os.path.join(log_dir, 'epoch_{epoch:03d}.h5'),
            save_freq=5 * steps_per_epoch,  # 每5个epoch保存一次
            save_weights_only=False,
            verbose=0
        ),
        #tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)#早停,耐心值为10,在10个epoch中性能没有提升会停止
    ]
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,  # 禁用默认输出
        callbacks=callbacks
    )
    plot_training_curve(os.path.join(log_dir,'train_log'))
    #model.save('ducknet_final.h5')