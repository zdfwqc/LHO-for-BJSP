import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d

def plot_training_logs(log_dir='.'):
    """
    读取并绘制log文件夹中的所有训练和测试损失曲线
    
    Args:
        log_dir (str): log文件夹的路径，默认为当前目录
    """
    # 获取所有训练损失文件
    train_files = glob.glob(os.path.join(log_dir, '*_train_loss.npy'))
    
    plt.figure(figsize=(12, 6))
    
    for train_file in train_files:
        # 获取模型名称
        model_name = os.path.basename(train_file).replace('_train_loss.npy', '')
        
        # 读取训练损失
        train_loss = np.load(train_file)
        
        # 尝试读取对应的测试损失
        test_file = os.path.join(log_dir, f'{model_name}_test_loss.npy')
        if os.path.exists(test_file):
            test_loss = np.load(test_file)
            
            # 创建插值函数
            x_old = np.linspace(0, 1, len(test_loss))
            x_new = np.linspace(0, 1, len(train_loss))
            f = interp1d(x_old, test_loss, kind='linear')
            test_loss_interpolated = f(x_new)
            
            plt.plot(test_loss_interpolated, label=f'{model_name} (test)', linestyle='--')
        
        plt.plot(train_loss, label=f'{model_name} (train)')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('LOSS')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'training_curves.png'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_training_logs(current_dir) 