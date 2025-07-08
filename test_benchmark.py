from LHOSolver import LHOSolver
import numpy as np
import os
import pandas as pd
from datetime import datetime
from network.deepSubmudularFunc import DSFDeepSet
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_instance(solver, instance_path, num_runs=10):
    """测试单个实例"""
    data = np.load(instance_path, allow_pickle=True)
    data = data[0]
    solver.reset(data)
    
    net_results = []
    
    # 神经网络结果
    net_results.append(solver.solve(data, model='net'))
    
    return np.mean(net_results)

def main():
    # 初始化求解器
    model = DSFDeepSet()
    solver = LHOSolver(numofMachines=5, k=7,net = model)

    solver.loadNet('model/dsf75.pth')
    
    # 测试实例列表
    instances = [f'la{i:02d}' for i in range(1, 16)]
    
    # 存储结果
    results = []
    
    # 测试每个实例
    for instance in instances:
        instance_path = f'benchmark/la/{instance}.npy'
        print(f"正在测试 {instance}...")
        
        try:
            net_result = test_instance(solver, instance_path)
            
            results.append({
                'Instance': instance,
                'Makespan': net_result
            })
            
            print(f"{instance} 测试完成:")
            print(f"神经网络结果: {net_result:.2f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"测试 {instance} 时出错: {str(e)}")
            continue
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果到CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'testlog/benchmark_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    
    # 打印总结
    print("\n测试总结:")
    print(f"平均Makespan: {df['Makespan'].mean():.2f}")
    print(f"详细结果已保存到: {csv_filename}")

if __name__ == '__main__':
    main() 