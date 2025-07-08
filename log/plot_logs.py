import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d

def plot_training_logs(log_dir='.', plot_test=True):
    """
    读取并绘制log文件夹中的所有训练和测试损失曲线
    
    Args:
        log_dir (str): log文件夹的路径，默认为当前目录
        plot_test (bool): 是否绘制测试损失曲线，默认为True
    """
    # 获取所有训练损失文件
    train_files = glob.glob(os.path.join(log_dir, '*_train_loss.npy'))
    
    # 创建主图和放大图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    for train_file in train_files:
        # 获取模型名称
        model_name = os.path.basename(train_file).replace('_train_loss.npy', '')
        
        # 读取训练损失
        train_loss = np.load(train_file)
        
        # 尝试读取对应的测试损失
        if plot_test:
            test_file = os.path.join(log_dir, f'{model_name}_test_loss.npy')
            if os.path.exists(test_file):
                test_loss = np.load(test_file)
                
                # 创建插值函数
                x_old = np.linspace(0, 1, len(test_loss))
                x_new = np.linspace(0, 1, len(train_loss))
                f = interp1d(x_old, test_loss, kind='linear')
                test_loss_interpolated = f(x_new)
                
                ax1.plot(test_loss_interpolated, label=f'{model_name} (test)', linestyle='--')
                ax2.plot(test_loss_interpolated, label=f'{model_name} (test)', linestyle='--')
        
        ax1.plot(train_loss, label=f'{model_name} (train)')
        ax2.plot(train_loss, label=f'{model_name} (train)')
    
    # 设置主图
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('LOSS')
    ax1.legend()
    ax1.grid(True)
    
    # 设置放大图
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.grid(True)
    
    # 设置放大图的y轴范围为主图最小值附近
    ymin = min([np.min(np.load(f)) for f in train_files])
    ax2.set_ylim(ymin*0.8, ymin*1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_training_logs(current_dir, plot_test=False) 