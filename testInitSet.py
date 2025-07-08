from LHOSolver import LHOSolver
from uniform_instance_gen import uni_instance_gen
from network.deepSubmudularFunc import DSFDeepSet
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 从文件加载数据或生成随机数据
    use_random = True # 设置是否使用随机生成的数据
    if use_random:
        num_instances = 100  # 生成100个随机实例
        
        # 测试神经网络模型不同初始化方式的结果
        all_net_random_init = []
        all_net_greedy_init = []
        all_net_cpgreedy_init = []
        all_net_grgc_init = []
        
        model =  DSFDeepSet()
        solver = LHOSolver(numofMachines=5, k=7,net=model)
        solver.loadNet('model/dsf75.pth')
        
        for i in range(num_instances):
            # 生成60个工件5个机器的随机实例,加工时间在1-99之间
            np.random.seed(i)
            times, machines = uni_instance_gen(10, 5, 1, 100)
            data = (times, machines)

            # 测试神经网络模型的不同初始化方式
            solver.reset(data)
            net_random_init = solver.solve(data, model='greedy', initChoose='random')
            
            solver.reset(data)
            net_greedy_init = solver.solve(data, model='greedy', initChoose='greedy')
            
            solver.reset(data)
            net_cpgreedy_init = solver.solve(data, model='greedy', initChoose='CPGreedy')

            solver.reset(data)
            net_grgc_init = solver.solve(data, model='greedy', initChoose='GRGC')
            
            all_net_random_init.append(net_random_init)
            all_net_greedy_init.append(net_greedy_init)
            all_net_cpgreedy_init.append(net_cpgreedy_init)
            all_net_grgc_init.append(net_grgc_init)
            
            print(f"\n实例 {i+1} 的结果:")
            print(f"神经网络+随机初始化: {net_random_init}")
            print(f"神经网络+贪心初始化: {net_greedy_init}")
            print(f"神经网络+CP贪心初始化: {net_cpgreedy_init}")
            print(f"神经网络+GRGC初始化: {net_grgc_init}")
            
        print("\n" + "="*60)
        print("神经网络模型不同初始化方式对比:")
        print("="*60)
        print("随机初始化平均结果:", np.mean(all_net_random_init))
        print("贪心初始化平均结果:", np.mean(all_net_greedy_init))
        print("CP贪心初始化平均结果:", np.mean(all_net_cpgreedy_init))
        print("GRGC初始化平均结果:", np.mean(all_net_grgc_init))
        
        # 计算标准差
        print("\n标准差分析:")
        print("随机初始化标准差:", np.std(all_net_random_init))
        print("贪心初始化标准差:", np.std(all_net_greedy_init))
        print("CP贪心初始化标准差:", np.std(all_net_cpgreedy_init))
        print("GRGC初始化标准差:", np.std(all_net_grgc_init))
        
    else:
        data = np.load('benchmark/la/la01.npy',allow_pickle=True)
        data = data[0]
        model =  DSFDeepSet(input_dim=15)
        model.load_state_dict(torch.load('model/dsf75.pth'))
        solver = LHOSolver(numofMachines=5, k=5,net=model)
        
        print("\n基准测试实例结果:")
        print("="*40)
        
        # 测试神经网络模型的不同初始化方式
        print("神经网络模型不同初始化方式对比:")
        print("-"*40)
        
        solver.reset(data)
        net_random_init = solver.solve(data, model='net', initChoose='random')
        print("随机初始化结果:", net_random_init)
        
        solver.reset(data)
        net_greedy_init = solver.solve(data, model='net', initChoose='greedy')
        print("贪心初始化结果:", net_greedy_init)
        
        solver.reset(data)
        net_cpgreedy_init = solver.solve(data, model='net', initChoose='CPGreedy')
        print("CP贪心初始化结果:", net_cpgreedy_init)

        solver.reset(data)
        net_grgc_init = solver.solve(data, model='net', initChoose='GRGC')
        print("GRGC初始化结果:", net_grgc_init)
