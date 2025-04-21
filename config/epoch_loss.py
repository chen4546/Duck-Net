import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
def plot_training_curve(train_log_path):
    train_file = os.path.join(train_log_path, 'train_loss.txt')
    output_path=os.path.join(train_log_path,'epoch_loss.png')
    loss_values=[]
    with open(train_file,'r') as f:
        for line in f:
            loss=float(line.strip())
            loss_values.append(loss)
    if not loss_values:
        print("训练日志为空，无法生成图表")
        return
    epochs=np.arange(1,len(loss_values)+1)
    #print(epochs)
    #print(loss_values)
    log_df=pd.DataFrame({'epoch': epochs, 'loss': loss_values})
    #print(log_df)
    # 动态获取轮次和损失范围
    max_epoch = log_df['epoch'].max()
    min_loss = log_df['loss'].min()
    max_loss = log_df['loss'].max()

    # 自动调整纵轴范围（留10%缓冲空间）
    loss_buffer = (max_loss - min_loss) * 0.1
    y_min = max(0, min_loss - loss_buffer)  # 确保不低于0
    y_max = max_loss + loss_buffer

    # 自动调整横轴刻度步长
    epoch_step = max(1, int(max_epoch / 5))  # 至少显示5个刻度

    # 动态调整平滑参数（基于数据长度）
    sigma = max(1.0, 30 / len(log_df))

    # 生成平滑曲线
    smooth_loss = gaussian_filter1d(log_df['loss'], sigma=sigma)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['loss'], label='train loss', linewidth=2.5, color='#d62728')
    plt.plot(log_df['epoch'], smooth_loss, label='smooth train loss', linewidth=2.5, linestyle='--',color='#007B00')

    # 设置动态坐标轴
    plt.xlim(0, max_epoch)
    plt.ylim(y_min, y_max)
    plt.xticks(range(0, max_epoch + 1, epoch_step))

    # 样式设置
    plt.xlabel("Epoch", fontsize=12, fontweight='bold')
    plt.ylabel("Loss", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    #功能测试
    plot_training_curve("log")