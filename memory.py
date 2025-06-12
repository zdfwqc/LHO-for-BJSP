import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import re

class Memory:
    def __init__(self, num_machines, capacity=100000):
        self.capacity = capacity
        self.num_machines = num_machines
        self.memory = []
        self.position = 0
        self.normalization_params = None
        
    def normalize(self, normalization_params=None):
        if len(self.memory) == 0:
            return
            
        # 如果提供了归一化参数，直接使用
        if normalization_params is not None:
            self.normalization_params = normalization_params
            normalized_memory = []
            for features, target in self.memory:
                normalized_target = (target - normalization_params['target_mean']) / normalization_params['target_std']
                normalized_memory.append((features, normalized_target))
            self.memory = normalized_memory
            return
            
        # 否则计算新的归一化参数
        all_targets = np.array([item[1] for item in self.memory])
        
        # 计算目标值的均值和标准差
        target_mean = np.mean(all_targets)
        target_std = np.std(all_targets)
        if target_std == 0:
            target_std = 1
        
        # 保存归一化参数
        self.normalization_params = {
            'target_mean': target_mean,
            'target_std': target_std
        }
        
        # 归一化数据，只归一化目标值
        normalized_memory = []
        for features, target in self.memory:
            # 归一化目标值
            normalized_target = (target - target_mean) / target_std
            normalized_memory.append((features, normalized_target))
            
        self.memory = normalized_memory
        
    def denormalize(self, normalized_value):
        if self.normalization_params is None:
            return normalized_value
            
        return normalized_value * self.normalization_params['target_std'] + self.normalization_params['target_mean']
        
    def normalize_value(self, original_value):
        return (original_value - self.normalization_params['target_mean']) / self.normalization_params['target_std']
        
    def denormalize_value(self, normalized_value):
        return normalized_value * self.normalization_params['target_std'] + self.normalization_params['target_mean']
        
    def push(self, jobs_features, utilization):
        features = np.array(jobs_features)
        
        # 存储数据
        if len(self.memory) < self.capacity:
            self.memory.append((features, utilization))
        else:
            # 随机选择要删除的位置
            remove_idx = np.random.randint(0, 100)
            # 删除选定位置的数据
            self.memory.pop(remove_idx)
            # 添加新数据
            self.memory.append((features, utilization))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size)
        features, utilizations = zip(*[self.memory[i] for i in indices])
        return torch.FloatTensor(features), torch.FloatTensor(utilizations)
    
    def __len__(self):
        return len(self.memory)

    @classmethod
    def _clean_filename(cls, filename):
        """
        清理文件名，移除非法字符
        
        参数:
        filename: 原始文件名
        
        返回:
        str: 清理后的文件名
        """
        # 移除Windows不允许的字符
        illegal_chars = r'[\\/:*?"<>|]'
        clean_name = re.sub(illegal_chars, '_', filename)
        return clean_name

    def save(self, filepath):
        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        clean_filename = self._clean_filename(filename)
        clean_filepath = os.path.join(dirname, clean_filename) if dirname else clean_filename
        
        # 检查文件是否存在,如果不存在则创建空文件
        if not os.path.exists(clean_filepath):
            open(clean_filepath, 'w').close()
            
        # 保存对象
        with open(clean_filepath, 'wb') as f:
            # 将numpy数组转换为列表，以确保可以正确序列化
            memory_data = [(features.tolist() if isinstance(features, np.ndarray) else features, 
                          utilization) for features, utilization in self.memory]
            pickle.dump({
                'capacity': self.capacity,
                'num_machines': self.num_machines,
                'memory': memory_data
            }, f)

    def split_memory(self, test_ratio=0.1):
        # 计算测试集大小
        test_size = int(len(self.memory) * test_ratio)

        # 随机选择测试集索引
        test_indices = set(np.random.choice(len(self.memory), test_size, replace=False))

        # 创建新的Memory实例
        train_memory = Memory(num_machines=self.num_machines, capacity=len(self.memory) - test_size)
        test_memory = Memory(num_machines=self.num_machines, capacity=test_size)

        # 分配数据
        for i, (features, target) in enumerate(self.memory):
            if i in test_indices:
                test_memory.memory.append((features, target))
            else:
                train_memory.memory.append((features, target))

        return train_memory, test_memory
            
    @classmethod
    def load(cls, filepath):
        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        clean_filename = cls._clean_filename(filename)
        clean_filepath = os.path.join(dirname, clean_filename) if dirname else clean_filename
        with open(clean_filepath, 'rb') as f:
            data = pickle.load(f)
        memory = cls(num_machines=data['num_machines'], capacity=data['capacity'])
        memory.memory = [(np.array(features) if isinstance(features, list) else features, 
                         utilization) for features, utilization in data['memory']]
        return memory 